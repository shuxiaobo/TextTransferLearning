#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/13
import torch
import torch.nn as nn
import torch.nn.functional as F
from senteval.utils import gather_rnnstate

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class BaseLineRNN(nn.Module):

    def __init__(self, args):
        super(BaseLineRNN, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.vocabulary_size = args.vocabulary_size

        bidirection = True
        self.embedding = args.embedding if args.embedding else nn.Embedding(args.vocabulary_size, args.embedding_size * 3)
        self.rnn = nn.GRU(self.embedding_size * 3, self.hidden_size, num_layers = 1, bias = False, bidirectional = bidirection, batch_first = True)
        self.dropout = nn.Dropout(p = self.args.keep_prob)

    def forward(self, x):
        x = LongTensor(x)
        mask = torch.where(x > 0, torch.ones_like(x, dtype = torch.float32), torch.zeros_like(x, dtype = torch.float32))
        x_embed = self.embedding(x)  # here can not use the mask while using the last hidden state.
        x_embed = self.dropout(x_embed)
        outputs, h = self.rnn(x_embed)
        outputs = gather_rnnstate(data = outputs, mask = mask)
        return outputs

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.all_weights.data)
        nn.init.xavier_uniform_(self.linear.weight.data)

    def loss(self, out, label):
        return F.cross_entropy(out, LongTensor(label))
