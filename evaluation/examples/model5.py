#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Desc: 
    @Author: shane 
    @Contact: iamshanesue@gmail.com
    @Software: PyCharm
    @Since: Python3.6
    @Date: 2018/11/27
    @All right reserved
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from senteval.utils import gather_rnnstate

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class NGramSumRNN(nn.Module):

    def __init__(self, args):
        super(NGramSumRNN, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.vocalbulary_size = args.vocabulary_size

        self.embedding = args.embedding if args.embedding else nn.Embedding(args.vocabulary_size, args.embedding_size)
        self.pos_embedding = nn.Embedding(5, self.embedding_size)
        self.pos_embedding2 = nn.Embedding(5, self.embedding_size)
        self.rnn1 = nn.GRU(self.embedding_size, hidden_size = self.hidden_size, num_layers = 1, bias = False, bidirectional = True,
                           batch_first = True)
        self.dropout = nn.Dropout(p = self.args.keep_prob)
        self.param = nn.Parameter(torch.randn(self.embedding_size * 3, self.embedding_size * 2))
        self.param2 = nn.Parameter(torch.randn(self.embedding_size * 2, self.embedding_size))

    def forward(self, x):
        x = LongTensor(x)
        mask = torch.where(x > 0, torch.ones_like(x, dtype = torch.float32), torch.zeros_like(x, dtype = torch.float32))
        x_embed = self.embedding(x)
        x_embed = self.dropout(x_embed)

        x_embed2 = self.embedding(torch.cat([torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda(), x[:, :-1]], -1))
        x_embed3 = self.embedding(torch.cat([x[:, 1:], torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda()], -1))
        x_embed4 = self.embedding(torch.cat([torch.zeros(size = [x.shape[0], 2], dtype = torch.long).cuda(), x[:, :-2]], -1))
        x_embed5 = self.embedding(torch.cat([x[:, 2:], torch.zeros(size = [x.shape[0], 2], dtype = torch.long).cuda()], -1))

        pos1 = self.pos_embedding(LongTensor([0, 1, 2]))
        pos2 = self.pos_embedding2(LongTensor([3, 0, 1, 2, 4]))

        ngram3 = F.softmax(torch.stack([x_embed2, x_embed, x_embed3], -2) * pos1, 2).max(-2)[0]
        ngram5 = F.softmax(torch.stack([x_embed4, x_embed2, x_embed, x_embed3, x_embed5], -2) * pos2, 2).max(-2)[0]
        # ngram3 = F.softmax(torch.sum(ngram3 * pos1, -1), -2).unsqueeze(-1) * ngram3
        # ngram5 = F.softmax(torch.sum(ngram5 * pos2, -1), -2).unsqueeze(-1) * ngram5

        # x_embed = torch.cat([ngram3.max(2)[0], ngram5.max(2)[0], x_embed], -1)
        # x_embed = torch.cat([x_embed, ngram3.sum(2).squeeze(2), ngram5.sum(2).squeeze(2)], -1)
        ngram3_s, ngram5_s = torch.split(torch.cat([ngram3, ngram5, x_embed], -1) @ self.param, self.embedding_size, -1)  #
        ngram3_s, ngram5_s = F.softmax(ngram3_s, -1), F.softmax(ngram5_s, -1)
        # ngram5_s = F.softmax(torch.cat([ngram5, x_embed], -1) @ self.param2, -1)  #
        x_embed = x_embed + ngram5_s * (F.tanh(ngram5) + ngram3_s * F.tanh(ngram3))
        outputs, (h, c) = self.rnn1(x_embed)
        # output_maxpooled, _ = torch.max(outputs, 1)
        output_maxpooled = gather_rnnstate(outputs, mask).squeeze(1)
        return F.dropout(output_maxpooled)
