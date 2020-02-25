# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .advanced_dropout import embedded_dropout

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class, dropout):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Invalid type for input_dims!'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)

        for i, n_hidden in enumerate(n_hiddens):
            l_i = i + 1
            layers['fc{}'.format(l_i)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(l_i)] = nn.ReLU()
            layers['drop{}'.format(l_i)] = nn.Dropout(dropout)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model.forward(input)

class MLP_Discriminator(nn.Module):
    def __init__(self, embed_dim, output_dim, hparams, use_cuda):
        super(MLP_Discriminator, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.hidden_state = hparams["hidden_state"]
        self.hidden_layers = hparams["hidden_layers"]
        self.hidden_dropout = hparams["hidden_dropout"]
        self.input_dropout = hparams["input_dropout"]
        self.bidirectional = hparams["bidirectional"]
        self.use_cuda = use_cuda

        self.mlp = MLP(embed_dim * 5, [self.hidden_state] * self.hidden_layers,
                       output_dim, self.hidden_dropout)
        self.dropout = nn.Dropout(self.input_dropout)
        if self.bidirectional:
            self.backward_mlp = MLP(embed_dim * 5, [self.hidden_state] * self.hidden_layers,
                                    output_dim, self.hidden_dropout)
            self.backward_dropout = nn.Dropout(self.input_dropout)

    def forward(self, s1, s2):
        inputs = torch.cat([s1, s2, s1 - s2, s1 * s2, torch.abs(s1 - s2)], -1)
        scores = self.mlp(self.dropout(inputs))
        if self.bidirectional:
            backward_inputs = torch.cat([s2, s1, s2 - s1, s1 * s2, torch.abs(s1 - s2)], -1)
            backward_scores = self.backward_mlp(self.backward_dropout(backward_inputs))
            scores = (scores + backward_scores) / 2
        return scores
