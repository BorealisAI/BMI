# Original work by Yang et al., shown here: https://github.com/zihangdai/mos/blob/master/model.py
# The original code is licensed under the license found in the
# LICENSE_MOS_ORIGINAL file in this directory
#
# Modifications are Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pickle
import time
import os
from .advanced_dropout import LockedDropout, embedded_dropout, WeightDrop
from .base_network import MLP_Discriminator
from torch.distributions import Categorical
import gc

def repackage_hidden(h):
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()

def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(1).expand(max_len, batch_size)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(0)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def compute_loss(logprob, target, length):
    logprob_flat = logprob.view(-1, logprob.size(-1))
    target_flat = target.view(-1)
    losses_flat = F.nll_loss(logprob_flat, target_flat, reduction='none')
    losses = losses_flat.view(*target.size())
    mask = _sequence_mask(length, target.size(0))
    losses = losses * mask.float()
    # loss = losses.sum() / length.float().sum()
    loss = losses.sum(0) / length.float().sum(0)
    return loss

def extract_last_hidden(output, length):
    batch_size = length.size(0)
    idx = (length - 1).unsqueeze(0).expand(batch_size, output.size(-1))
    idx = idx.unsqueeze(0)
    last_hidden = output.gather(0, Variable(idx))
    return last_hidden


class RNNModel(nn.Module):
    def __init__(self, vocab_size, hparams, use_cuda):
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = hparams['embed_size']
        self.hidden_size = hparams['hidden_size']
        self.last_hidden_size = hparams['last_hidden_size']
        self.num_layers = hparams['num_layers']
        self.dropout = hparams['dropout']
        self.dropouth = hparams['dropouth']
        self.dropouti = hparams['dropouti']
        self.dropoute = hparams['dropoute']
        self.dropoutl = hparams['dropoutl']
        self.wdrop = hparams['wdrop']
        self.n_experts = hparams['n_experts']
        self.tie_weights = hparams['tie_weights']
        self.use_cuda = use_cuda

        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(vocab_size, self.embed_size)
        self.rnns = [nn.LSTM(
            self.embed_size if x == 0 else self.hidden_size,
            self.hidden_size if x != self.num_layers - 1 else self.last_hidden_size,
            1) for x in range(self.num_layers)]
        if self.wdrop:
            self.rnns = [WeightDrop(
                rnn, ['weight_hh_l0'], dropout=self.wdrop) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)

        self.prior = nn.Linear(self.last_hidden_size,
                               self.n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(
            self.last_hidden_size, self.n_experts * self.embed_size), nn.Tanh())
        self.decoder = nn.Linear(self.embed_size, vocab_size)

        if self.tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('param size: {}'.format(size))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        hidden = []
        for x in range(self.num_layers - 1):
            hidden.append((Variable(torch.zeros(1, batch_size, self.hidden_size)),
                           Variable(torch.zeros(1, batch_size, self.hidden_size))))
        hidden.append((Variable(torch.zeros(1, batch_size, self.last_hidden_size)),
                       Variable(torch.zeros(1, batch_size, self.last_hidden_size))))
        return [(h.cuda(), c.cuda()) for h, c in hidden] if self.use_cuda else hidden

    def encode(self, sent, hidden, length):
        embed = embedded_dropout(self.encoder, sent,
                                 dropout=self.dropoute if self.training else 0)
        embed = self.lockdrop(embed, self.dropouti)

        if length is not None:
            length = np.array(length)
            sent_len_sorted, idx_sort = np.sort(length)[::-1], np.argsort(-length)
            sent_len_sorted = sent_len_sorted.copy()
            idx_unsort = np.argsort(idx_sort)
            idx_sort = torch.from_numpy(idx_sort).cuda() if self.use_cuda \
                else torch.from_numpy(idx_sort)
            idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.use_cuda \
                else torch.from_numpy(idx_unsort)

        raw_output = embed
        new_hidden = []
        raw_outputs = []
        outputs = []
        for i, rnn in enumerate(self.rnns):
            if length is not None:
                raw_output = raw_output.index_select(1, idx_sort)
                raw_output = nn.utils.rnn.pack_padded_sequence(raw_output, sent_len_sorted)

            raw_output, new_h = rnn(raw_output, hidden[i])

            if length is not None:
                raw_output = nn.utils.rnn.pad_packed_sequence(raw_output)[0]
                raw_output = raw_output.index_select(1, idx_unsort)
                new_h = tuple(h.index_select(1, idx_unsort) for h in new_h)

            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if i != self.num_layers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        return new_hidden, raw_output, raw_outputs, outputs

    def forward(self, sent, hidden, return_h=False, return_prob=False, encoding=False, length=None):
        batch_size = sent.size(1)

        hidden, raw_output, raw_outputs, outputs = self.encode(sent, hidden, length)

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        if encoding:
            return hidden, raw_outputs, outputs

        latent = self.latent(output)
        latent = self.lockdrop(latent, self.dropoutl)
        logit = self.decoder(latent.view(-1, self.embed_size))

        prior_logit = self.prior(output).contiguous().view(-1, self.n_experts)
        prior = F.softmax(prior_logit, -1)

        prob = F.softmax(logit.view(-1, self.vocab_size), -1)
        prob = prob.view(-1, self.n_experts, self.vocab_size)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(prob.add_(1e-8))
            model_output = log_prob

        model_output = model_output.view(-1, batch_size, self.vocab_size)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden


class LanguageModel:
    def __init__(self, train, valid, test, save_path, data, hparams):
        self.vocab_size = len(train.vocab)
        self.save_path = save_path
        self.data = data
        self.hparams = hparams
        self.num_epochs = hparams["num_epochs"]
        self.bsz1 = hparams["bsz1"]
        self.small_bsz1 = hparams["small_bsz1"]
        self.bsz2 = hparams["bsz2"]
        self.bszq = 3
        self.log_interval = hparams["log_interval"]
        self.lr = hparams["lr"]
        self.wdecay = hparams["wdecay"]
        self.single_gpu = hparams["single_gpu"]
        self.rml = hparams["rml"]
        self.contrastive1 = hparams["contrastive1"]
        self.contrastive2 = hparams["contrastive2"]
        self.contrastive2_rl = hparams["contrastive2_rl"]

        self.weight1 = hparams["weight1"]
        self.weight2 = hparams["weight2"]
        self.tau = hparams["tau"]
        self.geo_s_p = hparams["geo_s_p"]

        self.use_cuda = torch.cuda.is_available()

        self.train_dataloader = train
        self.valid_dataloader = valid
        self.test_dataloader = test

        self.c_lr = c_lr = hparams["c_lr"]
        self.c_wdecay = c_wdecay = hparams["c_wdecay"]
        mlp_params = hparams["mlp_params"]
        repr_dim = hparams["last_hidden_size"] + 2 * hparams["hidden_size"]

        self.lm = RNNModel(self.vocab_size, hparams, self.use_cuda)
        self.discriminator = MLP_Discriminator(
            repr_dim, 1, mlp_params, self.use_cuda)

        if self.single_gpu:
            self.parallel_lm = self.lm
            self.parallel_discriminator = self.discriminator
        else:
            self.parallel_lm = nn.DataParallel(self.lm, dim=1)
            self.parallel_discriminator = nn.DataParallel(self.discriminator)

        if self.use_cuda:
            self.parallel_lm.cuda()
            self.parallel_discriminator.cuda()

        self.optimizer = optim.SGD(
            self.lm.parameters(), lr=self.lr, weight_decay=self.wdecay)
        params = list(self.discriminator.parameters())
        self.c_optimizer = optim.Adam(params, lr=c_lr, weight_decay=c_wdecay)

    def _variable(self, data, transpose=True):
        data = np.array(data)
        data = Variable(torch.from_numpy(data))
        if transpose:
            data = data.transpose(1, 0)
        return data.cuda() if self.use_cuda else data

    def logging(self, s, log_name='log.txt', print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.save_path, log_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def init(self):
        def init_weights(model):
            if type(model) in [nn.Linear]:
                nn.init.xavier_normal_(model.weight.data)
            elif type(model) in [nn.LSTM, nn.GRU]:
                nn.init.xavier_normal_(model.weight_hh_l0)
                nn.init.xavier_normal_(model.weight_ih_l0)

        self.discriminator.apply(init_weights)

    def generate_text_rl(self, hidden, bsz, nbatch, tokens=None, training=True, self_critical=True):
        if not training:
            self.lm.eval()

        generated_text = []
        if tokens is None:
            tokens = np.random.randint(self.vocab_size, size=(1, bsz))
            # tokens = np.ones((1, bsz), dtype=int)
            tokens = Variable(torch.from_numpy(tokens))

        log_probs = torch.zeros(nbatch, bsz)
        if self.use_cuda:
            tokens = tokens.cuda()
            log_probs = log_probs.cuda()

        for i in range(nbatch):
            prob, hidden = self.parallel_lm(tokens, hidden, False, True)
            prob = prob.squeeze()
            m = Categorical(prob)

            tokens = m.sample().detach()
            _, tokens[-1] = prob[-1].max(0)  # greedy

            token_log_prob = m.log_prob(tokens)
            tokens = tokens.unsqueeze(0)
            generated_text.append(tokens)
            log_probs[i, :] += token_log_prob

        text = torch.cat(generated_text, 0).contiguous()
        if training:
            self.lm.train()

        return text, hidden, log_probs

    def generate_text(self, hidden, bsz, nbatch, tokens=None, training=True):
        self.lm.eval()
        generated_text = []
        # hidden = self.lm.init_hidden(bsz)
        if tokens is None:
            tokens = np.random.randint(self.vocab_size, size=(1, bsz))
            # tokens = np.ones((1, bsz), dtype=int)
            tokens = Variable(torch.from_numpy(tokens))
        if self.use_cuda:
            tokens = tokens.cuda()
        with torch.no_grad():
            for i in range(nbatch):
                prob, hidden = self.parallel_lm(tokens, hidden, False, True)
                tokens = torch.multinomial(prob.squeeze(0), 1)
                tokens = tokens.transpose(1, 0)
                generated_text.append(tokens)

        text = torch.cat(generated_text, 0)
        # text = torch.cat(generated_text, 0).cpu().numpy()
        # path = os.path.join(self.save_path, "generated_text.npy")
        # np.save(path, np.transpose(text))
        if training:
            self.lm.train()
        return text, hidden

    def encode(self, text, hidden, length=None):
        hidden, raw_outputs, outputs = self.parallel_lm(text, hidden, encoding=True)
        repr1 = torch.max(outputs[0], 0)[0]
        repr2 = torch.max(outputs[1], 0)[0]
        repr3 = torch.max(raw_outputs[2], 0)[0]
        return torch.cat([repr1, repr2, repr3], -1), hidden

    def train(self, epoch, log_name="log.txt"):
        self.lm.train()
        self.discriminator.train()

        total_loss = 0
        hidden = [self.lm.init_hidden(self.small_bsz1) for _ in range(self.bsz1 // self.small_bsz1)]
        ini_hidden = self.lm.init_hidden(self.bsz2)
        ini_hiddenq = self.lm.init_hidden(self.bszq)
        start_time = time.time()
        nbatch = int(len(self.train_dataloader.indices) / self.bsz1 / 70)
        step = 0

        for batch in self.train_dataloader.fetch_batches(
                self.bsz1, self.bsz2, data=self.data, geo_s_p=self.geo_s_p):
            source, target, text, text_len, pos_nxt, neg_nxt, rml, rml_len, rml_t, geo_ps = batch
            source = self._variable(source)
            target = self._variable(target)
            text = self._variable(text)
            pos_nxt = self._variable(pos_nxt)
            neg_nxt = self._variable(neg_nxt)

            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = lr * \
                source.size(0) / 70  # adjust based on seq len
            self.optimizer.zero_grad()
            self.c_optimizer.zero_grad()

            start, end, s_id = 0, self.small_bsz1, 0
            while start < self.bsz1:
                cur_source, cur_target = source[:, start:end], target[:, start:end].contiguous()
                log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = self.parallel_lm(
                    cur_source, hidden[s_id], True)
                hidden[s_id] = repackage_hidden(hidden[s_id])

                raw_loss = F.nll_loss(
                    log_prob.view(-1, self.vocab_size), cur_target.view(-1))
                lm_loss = raw_loss
                lm_loss = lm_loss + sum(2.0 * h.pow(2).mean() for h in dropped_rnn_hs[-1:])
                lm_loss = lm_loss + sum(1.0 * (h[1:] - h[:-1]).pow(2).mean() for h in rnn_hs[-1:])
                lm_loss *= self.small_bsz1 / self.bsz1
                total_loss += raw_loss.data.cpu() * self.small_bsz1 / self.bsz1

                s_id += 1
                start = end
                end = start + self.small_bsz1

                lm_loss.backward()
                gc.collect()

            contrastive_loss1 = 0.
            contrastive_loss2 = 0.
            if self.contrastive1:
                repr1, h1 = self.encode(text, ini_hidden, text_len)
                repr2, h2 = self.encode(pos_nxt, ini_hidden)
                repr3, h3 = self.encode(neg_nxt, ini_hidden)
                pos_scores = self.parallel_discriminator(repr1, repr2)
                neg_scores = self.parallel_discriminator(repr1, repr3)

                contrastive_loss1 += (F.softplus(-pos_scores)).mean()
                contrastive_loss1 += (F.softplus(neg_scores)).mean()
                contrastive_loss1 *= self.weight1

                if self.rml:
                    rml = self._variable(rml)
                    rml_len = self._variable(rml_len, False)
                    rml_t = self._variable(rml_t)
                    geo_ps = self._variable(geo_ps, False)
                    # rml_t, _ = self.generate_text(h1, self.bsz2, 40)
                    # tokens = np.ones((1, self.bsz2), dtype=int)
                    # tokens = Variable(torch.from_numpy(tokens))
                    # if self.use_cuda:
                    #     tokens = tokens.cuda()
                    # rml = torch.cat([tokens, rml_t[:-1, :]], 0)

                    rml_log_prob, _, raw_outputs, outputs = self.parallel_lm(
                        rml, h1, True)

                    # nll = F.nll_loss(
                    #     rml_log_prob.view(-1, self.vocab_size), rml_t.view(-1), reduction="none")
                    # nll = nll.reshape(*rml.size())
                    nll = compute_loss(rml_log_prob, rml_t, rml_len)

                    r1 = torch.max(outputs[0], 0)[0]
                    r2 = torch.max(outputs[1], 0)[0]
                    r3 = torch.max(raw_outputs[2], 0)[0]
                    repr4 = torch.cat([r1, r2, r3], -1)
                    scores = self.parallel_discriminator(
                        repr1, repr4) - pos_scores.detach()
                    scores = scores - scores.max()
                    scores /= self.tau

                    un_r_p = torch.exp(scores) / geo_ps[:, None]
                    # un_r_p = torch.exp(scores)
                    w = un_r_p / un_r_p.sum()
                    w = w.detach().t()

                    rml_loss = (w * nll).sum()

                    contrastive_loss1 += self.weight2 * rml_loss

                contrastive_loss1.backward()

            if self.contrastive2:
                bsz2 = self.bsz2

                if self.contrastive2_rl:
                    T = 15  # memory issue if too large
                    self_critical = True  # self critical baseline for variance reduction

                    if self_critical:
                        bsz2 += 1

                    t1, h1, tlp1 = self.generate_text_rl(
                        ini_hiddenq, bsz2, T, self_critical=self_critical)
                    h1 = [(h[0].detach(), h[1].detach())
                          for h in h1]  # memory issue if not detached
                    t2, _, tlp2 = self.generate_text_rl(
                        h1, bsz2, T, t1[-1:, :])
                    t3, _, tlp3 = self.generate_text_rl(ini_hiddenq, bsz2, T)
                else:
                    T = 40
                    t1, h1 = self.generate_text(ini_hiddenq, bsz2, T)
                    t2, h2 = self.generate_text(h1, bsz2, T, t1[-1:, :])
                    t3, h3 = self.generate_text(ini_hiddenq, bsz2, T)

                t1 = t1.detach()
                t2 = t2.detach()
                t3 = t3.detach()

                repr1, _ = self.encode(t1, ini_hiddenq)
                repr2, _ = self.encode(t2, ini_hiddenq)
                repr3, _ = self.encode(t3, ini_hiddenq)

                pos_scores = self.parallel_discriminator(repr1, repr2)
                neg_scores = self.parallel_discriminator(repr1, repr3)

                pos_losses = F.softplus(-pos_scores)
                neg_losses = F.softplus(neg_scores)

                contrastive_loss2 += pos_losses.mean()
                contrastive_loss2 += neg_losses.mean()

                if self.contrastive2_rl:
                    p_reward = pos_losses.detach().t()
                    n_reward = neg_losses.detach().t()

                    if self_critical:

                        p_baseline = p_reward[:, -1]
                        n_baseline = n_reward[:, -1]

                        p_reward = p_reward[:, :-1] - p_baseline
                        n_reward = n_reward[:, :-1] - n_baseline

                        tlp1 = tlp1[:, :-1]
                        tlp2 = tlp2[:, :-1]
                        tlp3 = tlp3[:, :-1]

                    pg_loss = - (tlp1 * p_reward).mean()
                    pg_loss += - (tlp2 * p_reward).mean()
                    pg_loss += - (tlp1 * n_reward).mean()
                    pg_loss += - (tlp3 * n_reward).mean()

                    contrastive_loss2 += pg_loss

                contrastive_loss2 *= self.weight2
                contrastive_loss2.backward()

            # loss = lm_loss + self.weight1 * contrastive_loss1
            # loss = loss + self.weight2 * contrastive_loss2
            # loss.backward()

            nn.utils.clip_grad_norm_(self.lm.parameters(), 0.25)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.25)
            self.optimizer.step()
            self.optimizer.param_groups[0]['lr'] = lr
            self.c_optimizer.step()

            if step % self.log_interval == 0 and step > 0:
                cur_loss = total_loss.item() / self.log_interval
                elapsed = time.time() - start_time
                self.logging('| epoch {:2d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | '
                             'loss {:2.2f} | ppl {:3.2f}'.format(
                                 epoch, step, nbatch, elapsed * 1000 / self.log_interval,
                                 cur_loss, np.exp(cur_loss)),
                             log_name)
                gc.collect()
                total_loss = 0
                start_time = time.time()

            step += 1

    def evaluate(self, dataloader, bsz1, bsz2):
        self.lm.eval()
        self.discriminator.eval()

        total_loss = 0
        total_acc1 = 0
        total_acc2 = 0
        total_length = 0
        total_batch = 0
        hidden = self.lm.init_hidden(bsz1)
        ini_hidden = self.lm.init_hidden(bsz2)
        # ini_hiddenq = self.lm.init_hidden(self.bszq)
        with torch.no_grad():
            for batch in dataloader.fetch_batches(bsz1, bsz2, True):
                source, target, text, _, pos_nxt, neg_nxt, _, _, _, _ = batch
                source = self._variable(source)
                target = self._variable(target)
                text = self._variable(text)
                pos_nxt = self._variable(pos_nxt)
                neg_nxt = self._variable(neg_nxt)

                log_prob, hidden = self.parallel_lm(source, hidden)
                hidden = repackage_hidden(hidden)

                repr1, _ = self.encode(text, ini_hidden)
                repr2, _ = self.encode(pos_nxt, ini_hidden)
                repr3, _ = self.encode(neg_nxt, ini_hidden)

                pos_scores = self.parallel_discriminator(repr1, repr2)
                neg_scores = self.parallel_discriminator(repr1, repr3)
                total_acc1 += (pos_scores > neg_scores).float().mean()

                # t1, h1 = self.generate_text(ini_hiddenq, self.bszq, 40, training=False)
                # t2, h2 = self.generate_text(h1, self.bszq, 40, t1[-1:, :], training=False)
                # t3, h3 = self.generate_text(ini_hiddenq, self.bszq, 40, training=False)
                # t1 = t1.detach()
                # t2 = t2.detach()
                # t3 = t3.detach()
                # repr1, _ = self.encode(t1, ini_hiddenq)
                # repr2, _ = self.encode(t2, ini_hiddenq)
                # repr3, _ = self.encode(t3, ini_hiddenq)

                # pos_scores = self.parallel_discriminator(repr1, repr2)
                # neg_scores = self.parallel_discriminator(repr1, repr3)
                # total_acc2 += (pos_scores > neg_scores).float().mean()

                loss = F.nll_loss(
                    log_prob.view(-1, self.vocab_size), target.view(-1))
                total_loss += loss.data.cpu() * source.size(1)
                total_length += source.size(1)
                total_batch += 1
        return total_loss.item() / total_length, total_acc1 / total_batch, total_acc2 / total_batch

    def fit(self):
        best_valid_loss = np.inf
        best_valid_acc1 = 0.0
        best_valid_acc2 = 0.0
        best_valid_epoch = 0
        stored_losses = []
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch)
            if 't0' in self.optimizer.param_groups[0]:
                tmp = {}
                for prm in self.lm.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = self.optimizer.state[prm]['ax'].clone()

                val_loss, val_acc1, val_acc2 = self.evaluate(
                    self.valid_dataloader, 10, self.bsz2)
                if val_loss < best_valid_loss:
                    self.save(self.save_path)
                    best_valid_loss = val_loss
                    best_valid_epoch = epoch
                    best_valid_acc1 = val_acc1
                    best_valid_acc2 = val_acc2
                for prm in self.lm.parameters():
                    prm.data = tmp[prm].clone()
                if (len(stored_losses) > 5) and (val_loss > min(stored_losses[:-5])):
                    break
            else:
                val_loss, val_acc1, val_acc2 = self.evaluate(
                    self.valid_dataloader, 10, self.bsz2)
                if val_loss < best_valid_loss:
                    self.save(self.save_path)
                    best_valid_loss = val_loss
                    best_valid_epoch = epoch
                    best_valid_acc1 = val_acc1
                    best_valid_acc2 = val_acc2

                if ('t0' not in self.optimizer.param_groups[0]) and (len(stored_losses) > 10) \
                        and (val_loss > min(stored_losses[:-10])):
                    self.logging(
                        '======================= Switching! =======================')
                    self.save_path = self.save_path + '_switched'
                    if not os.path.exists(self.save_path):
                        os.mkdir(self.save_path)
                    self.optimizer = torch.optim.ASGD(self.lm.parameters(), t0=0, lambd=0.,
                                                      lr=self.lr, weight_decay=self.wdecay)
#                     if not self.contrastive2:
#                         self.contrastive2 = True
#                     else:
#                         self.logging('Switching!')
#                         self.optimizer = torch.optim.ASGD(self.lm.parameters(), t0=0, lambd=0.,
#                                                           lr=self.lr, weight_decay=self.wdecay)

                    stored_losses = []
            stored_losses.append(val_loss)

            self.logging('-' * 75)
            self.logging('| end of epoch {:2d} | time {:5.2f}s | valid loss {:2.2f} | '
                         'valid ppl {:3.2f} | valid acc {:.4f}, {:.4f}'.format(
                             epoch, (time.time() - epoch_start_time),
                             val_loss, np.exp(val_loss), val_acc1, val_acc2))
            self.logging('-' * 75)

        return best_valid_epoch, best_valid_loss, best_valid_acc1, best_valid_acc2

    def finetune(self):
        best_valid_loss = np.inf
        best_valid_acc1 = 0.0
        best_valid_acc2 = 0.0
        best_valid_epoch = 0
        stored_losses = []
        self.optimizer = torch.optim.ASGD(self.lm.parameters(), t0=0, lambd=0.,
                                          lr=self.lr, weight_decay=self.wdecay)
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch, "finetune_log.txt")
            tmp = {}
            for prm in self.lm.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = self.optimizer.state[prm]['ax'].clone()
            val_loss, val_acc1, val_acc2 = self.evaluate(
                self.valid_dataloader, 10, self.bsz2)
            if val_loss < best_valid_loss:
                self.save(self.save_path, True)
                best_valid_loss = val_loss
                best_valid_acc1 = val_acc1
                best_valid_acc2 = val_acc2
                best_valid_epoch = epoch
            for prm in self.lm.parameters():
                prm.data = tmp[prm].clone()
            if (len(stored_losses) > 5) and (val_loss > min(stored_losses[:-5])):
                break
            stored_losses.append(val_loss)

            self.logging('-' * 75, 'finetune_log.txt')
            self.logging('| end of epoch {:2d} | time {:5.2f}s | valid loss {:2.2f} | '
                         'valid ppl {:3.2f} | valid acc {:.4f}, {:.4f}'.format(
                             epoch, (time.time() - epoch_start_time),
                             val_loss, np.exp(val_loss), val_acc1, val_acc2), 'finetune_log.txt')
            self.logging('-' * 75, 'finetune_log.txt')

        return best_valid_epoch, best_valid_loss, best_valid_acc1, best_valid_acc2

    def save(self, path, finetune=False):
        self.logging('saving {}'.format(path), "finetune_log.txt" if finetune else "log.txt")

        model_path = os.path.join(path, "model.pt")
        optimizer_path = os.path.join(path, "optimizer.pt")
        discriminator_path = os.path.join(path, "discriminator.pt")
        c_optimizer_path = os.path.join(path, "c_optimizer.pt")
        if finetune:
            model_path = os.path.join(path, "ft_model.pt")
            optimizer_path = os.path.join(path, "ft_optimizer.pt")
            discriminator_path = os.path.join(path, "ft_discriminator.pt")
            c_optimizer_path = os.path.join(path, "ft_c_optimizer.pt")

        torch.save(self.lm.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
        torch.save(self.c_optimizer.state_dict(), c_optimizer_path)
        with open(os.path.join(path, "hparams.pkl"), "wb") as f:
            pickle.dump(self.hparams, f, -1)

    def load(self, path, finetune=False, dis=True):
        model_path = os.path.join(path, "model.pt")
        optimizer_path = os.path.join(path, "optimizer.pt")
        discriminator_path = os.path.join(path, "discriminator.pt")
        c_optimizer_path = os.path.join(path, "c_optimizer.pt")
        if finetune:
            model_path = os.path.join(path, "ft_model.pt")
            optimizer_path = os.path.join(path, "ft_optimizer.pt")
            discriminator_path = os.path.join(path, "ft_discriminator.pt")
            c_optimizer_path = os.path.join(path, "ft_c_optimizer.pt")

        self.lm.load_state_dict(torch.load(model_path))
        optimizer_state = torch.load(optimizer_path)
        if 't0' in optimizer_state['param_groups'][0]:
            self.optimizer = optim.ASGD(
                self.lm.parameters(), lr=self.lr, t0=0., lambd=0., weight_decay=self.wdecay)
        else:
            self.optimizer = optim.SGD(
                self.lm.parameters(), lr=self.lr, weight_decay=self.wdecay)
        self.optimizer.load_state_dict(optimizer_state)

        if dis:
            self.discriminator.load_state_dict(torch.load(discriminator_path))
            params = list(self.discriminator.parameters())
            self.c_optimizer = optim.Adam(
                params, lr=self.c_lr, weight_decay=self.c_wdecay)
            self.c_optimizer.load_state_dict(torch.load(c_optimizer_path))

        if self.single_gpu:
            self.parallel_lm = self.lm
            self.parallel_discriminator = self.discriminator
        else:
            self.parallel_lm = nn.DataParallel(self.lm, dim=1)
            self.parallel_discriminator = nn.DataParallel(self.discriminator)

        if self.use_cuda:
            self.parallel_lm.cuda()
            self.parallel_discriminator.cuda()
