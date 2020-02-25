# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
from collections import defaultdict, OrderedDict
import numpy as np

class Vocabulary:
    UNK = 0
    EOS = 1

    def __init__(self, vocab_size):
        self.word2idx = OrderedDict()
        self.word2idx['<unk>'] = self.UNK
        self.word2idx['<eos>'] = self.EOS
        self.idx2word = ['<unk>', '<eos>']
        self.vocab_size = vocab_size

    def __getitem__(self, word):
        idx = self.word2idx.get(word, self.UNK)
        return idx if idx < self.vocab_size else self.UNK

    def to_word(self, idx):
        return self.idx2word[idx]

    def __len__(self):
        return min(len(self.idx2word), self.vocab_size)

    def build(self, sents):
        wordcount = defaultdict(int)
        for sent in sents:
            words = sent.split()
            for w in words:
                if w in ['<unk>', '<eos>']:
                    continue
                wordcount[w] += 1
        sorted_words = sorted(wordcount, key=wordcount.get, reverse=True)

        for idx, word in enumerate(sorted_words):
            self.word2idx[word] = idx + 2
        self.idx2word = list(self.word2idx.keys())

    def load(self, path):
        with open(path, 'rb') as f:
            self.word2idx = pickle.load(f)
            self.idx2word = list(self.word2idx.keys())

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.word2idx, f, -1)

class DataLoader:
    def __init__(self, file_path, vocab_path, max_len_delta=40, mode='sent',
                 bptt=70, vocab_size=100000, max_len=40):
        self.max_len_delta = max_len_delta
        self.mode = mode
        self.bptt = bptt
        self.max_len = max_len
        sents = self.get_all_sentences(file_path)
        self.vocab = Vocabulary(vocab_size)
        try:
            self.vocab.load(vocab_path)
            print("Using cached vocabulary at {}".format(vocab_path))
        except Exception:
            print("Unable to load from cached, building fresh...")
            self.vocab.build(sents)
            print("Got {} unique words".format(len(self.vocab.idx2word)))
            self.vocab.save(vocab_path)
            print("Saving vocabulary at {}".format(vocab_path))
        self.sent_lens = [min(len(sent.split()) + 1, max_len)
                          for sent in sents]
        self.sents = [self.tokenize_sent(sent) for sent in sents]
        self.raw_sents = sents
        self.indices = self.tokenize_corpus(file_path)
        self.num_sent = len(self.sents)
        self.num_indices = len(self.indices)

    def get_all_sentences(self, file_path):
        sents = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                sents.append(line)
        return sents

    def tokenize_corpus(self, file_path):
        indices = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                words = line.split() + ['<eos>']
                for word in words:
                    indices.append(self.vocab[word])
        return indices

    def tokenize_sent(self, sent, shift=0):
        words = sent.split()[shift:]
        indices = [self.vocab[w]
                   for w in words][:self.max_len - 1] + [self.vocab.EOS]
        indices += [self.vocab.UNK] * (self.max_len - len(indices))
        return indices

    def get_tokenized_blocks(self, indices, seq_len):
        n = len(indices)
        i = 0
        blocks = []
        while i + seq_len <= n:
            blocks.append(indices[i:i + seq_len])
            i += seq_len
        return blocks

    def sample_sequence_length(self):
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, bptt + self.max_len_delta)
        return seq_len

    def fetch_batches(self, bsz1, bsz2, evaluation=False,
                      data=None, seq_len=None, geo_s_p=.3):
        if data is None:
            nbatch = len(self.indices) // bsz1
            data = self.indices[:nbatch * bsz1]
            data = np.array(data).reshape(bsz1, nbatch)
        else:
            nbatch = data.shape[1]

        i = 0
        while i < nbatch - 1 - 1:
            if evaluation:
                seq_len = seq_len if seq_len else self.bptt
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
                seq_len = int(min(seq_len, bptt + self.max_len_delta))
            seq_len = min(seq_len, nbatch - 1 - i)
            source = data[:, i:i + seq_len]
            target = data[:, i + 1:i + 1 + seq_len]

            text = []
            text_len = []
            pos_nxt = []
            neg_nxt = []

            rml = []
            rml_len = []
            rml_t = []
            geo_ps = []

            for _ in range(bsz2):
                if self.mode == 'sent':
                    idx = np.random.randint(self.num_sent - 1)
                    t1 = self.sents[idx]
                    t1_len = self.sent_lens[idx]
                    t2 = self.sents[idx + 1]

                    random_idx = np.random.randint(self.num_sent - 1)
                    while (random_idx - idx == 0) or (random_idx - idx == 1):
                        random_idx = np.random.randint(self.num_sent - 1)
                    t3 = self.sents[random_idx]

                    geo_idx = min(np.random.geometric(p=geo_s_p) - 1,
                                  self.num_sent - idx - 2)
                    t4 = self.sents[idx + geo_idx + 1]
                    t4_len = self.sent_lens[idx + geo_idx + 1]
                    t5 = self.tokenize_sent(self.raw_sents[idx + geo_idx + 1], shift=1)
                else:
                    idx = np.random.randint(self.num_indices - self.max_len * 2)
                    t1 = self.indices[idx:idx + self.max_len]
                    t1_len = self.max_len
                    t2 = self.indices[idx + self.max_len:idx + self.max_len * 2]

                    random_idx = np.random.randint(
                        self.num_indices - self.max_len)
                    while (random_idx - idx >= -self.max_len) and \
                            (random_idx - idx <= self.max_len * 2):
                        random_idx = np.random.randint(
                            self.num_indices - self.max_len)
                    t3 = self.indices[random_idx:random_idx + self.max_len]

                    geo_idx = min(np.random.geometric(p=geo_s_p) - 1,
                                  self.num_indices - idx - self.max_len * 2 - 1)
                    t4 = self.indices[idx + self.max_len +
                                      geo_idx:idx + self.max_len * 2 + geo_idx]
                    t4_len = self.max_len
                    t5 = self.indices[idx + self.max_len + 1 +
                                      geo_idx:idx + self.max_len * 2 + geo_idx + 1]

                text.append(t1)
                text_len.append(t1_len)
                pos_nxt.append(t2)
                neg_nxt.append(t3)
                rml.append(t4)
                rml_len.append(t4_len)
                rml_t.append(t5)
                geo_ps.append(np.float32(
                    (1. - geo_s_p) ** (geo_idx + 1) * geo_s_p))

            i += seq_len

            yield source, target, text, text_len, pos_nxt, neg_nxt, rml, rml_len, rml_t, geo_ps
