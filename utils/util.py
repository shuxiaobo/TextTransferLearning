#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/28
import logging
import numpy as np
from datasets import *
from collections import Counter

logger = logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(filename)s[line:%(lineno)d]： %(message)s', datefmt = '%Y-%m-%d %I:%M:%S')


def prepare_dictionary(data, dict_path, exclude_n = 10, max_size = None):
    """
    generate the data word2id dict. sorted by word frequence.
    write file with format : word \t times \n

    Note: for chinese we only receipt the segmented data.
    :param data:
        list data : [['one', 'two'...]...]
    :param dict_path: path to save the result dict
    :param exclude_n: exclude top n word
    :param max_size: dict max size, default is all, then we could filter the dict when load the word2id file.
    :return:
    """
    word2id = dict()
    for i, d in enumerate(data):
        for j, s in enumerate(d):
            if s not in word2id.keys():
                word2id.setdefault(s, 0)
            word2id[s] = word2id[s] + 1
    c = Counter(word2id)
    max_size = max_size if max_size else len(c) - exclude_n
    rs = [(PAD, PAD_ID), (UNKNOW, UNKNOW_ID)] + c.most_common(max_size + exclude_n)[exclude_n:]
    word2id = {k[0]: v for v, k in enumerate(rs)}
    with open(dict_path, mode = 'w+', encoding = 'utf-8') as f:
        for d in rs:
            f.write(d[0] + '\t' + d[1] + '\n')
    return word2id


def collect_fn(data):
    """
    The common collect fun for data loader
    return the collect data which should be processed
    :param data:
    :return:
    """
    return data


def accuracy(out, label):
    return np.sum(np.equal(np.argmax(out, axis = -1), label))
