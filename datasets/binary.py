'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA...
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import logging
from utils.util import logger
from torch.utils.data import Dataset
from utils.util import prepare_dictionary
from tensorflow.python.keras.preprocessing import sequence


class BinaryClassifierEval(Dataset):
    def __init__(self, args, num_class = 2, seed = 1111, filename = None):
        """

        :param args:
        :param num_class: result class number
        :param seed: random seed
        :param filename: train | valid | test filename, default is train
        """
        self.seed = seed
        self.args = args
        self.num_class = num_class
        self.max_len = 0
        filename = filename if filename else args.train_file

        self.data_x, self.data_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, filename))

        self.n_samples = len(self.data_x)
        self.word_file = os.path.join(args.tmp_dir, self.__class__.__name__, args.word_file)
        if os.path.isfile(self.word_file) and os.path.getsize(self.word_file) > 0:
            self.word2id = self.get_word_index(self.word_file)
        else:
            self.word2id = self.prepare_dict(self.word_file)

    def load_file(self, fpath):
        """
        load the data file with format : x \t y
        Note: the data_y should be the sparse id, and start from 0.
        for example if you have 3 class, the id must range in (0, 1, 2)

        :param fpath: file path
        :return: data_x, data_y
        """
        with io.open(fpath, 'r', encoding = 'utf-8') as f:
            data_x = list()
            data_y = list()
            for line in f.read().splitlines():
                line = line.strip().split(' ')
                data_x.append(line[:-1])
                data_y.append(int(line[-1]))
                self.max_len = len(line[:-1]) if len(line[:-1]) > self.max_len else self.max_len
        return data_x, data_y

    def prepare_dict(self, file_name):
        logger("Prepare the dictionary for the {}...".format(self.__class__.__name__))
        word2id = prepare_dictionary(data = self.data_x, dict_path = file_name, exclude_n = self.args.skip_top, max_size = self.args.num_words)
        logger("Word2id size : %d" % len(word2id))
        return word2id

    def get_word_index(self, path = None):
        if not path:
            path = self.args.tmp_dir + self.__class__.__name__ + self.args.word_file
        word2id = dict()
        with open(path, mode = 'r', encoding = 'utf-8') as f:
            for l in f:
                word2id.setdefault(l.strip(), len(word2id))
        logger('Word2id size : %d' % len(word2id))
        return word2id

    @staticmethod
    def batchfy_fn(data):
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        max_len = max(map(len, x))
        return sequence.pad_sequences(x, maxlen = max_len, padding = 'post'), y

    def __getitem__(self, index):
        result = [self.word2id[d] if self.word2id.get(d) else self.word2id['_<UNKNOW>'] for d in self.data_x[index]]
        return result, self.data_y[index]

    def __len__(self):
        return self.n_samples


class CR(BinaryClassifierEval):
    def __init__(self, args, seed = 1111, filename = None):
        logging.debug('***** Task : ' + self.__class__.__name__ + ' *****\n\n')
        super(self.__class__, self).__init__(args, seed, filename)


class MR(BinaryClassifierEval):
    def __init__(self, args, seed = 1111, filename = None):
        logging.debug('***** Task : ' + self.__class__.__name__ + ' *****\n\n')
        super(self.__class__, self).__init__(args, seed, filename)


class SUBJ(BinaryClassifierEval):
    def __init__(self, args, seed = 1111, filename = None):
        logging.debug('***** Task : ' + self.__class__.__name__ + ' *****\n\n')
        super(self.__class__, self).__init__(args, seed, filename)


class MPQA(BinaryClassifierEval):
    def __init__(self, args, seed = 1111, filename = None):
        logging.debug('***** Task : ' + self.__class__.__name__ + ' *****\n\n')
        super(self.__class__, self).__init__(args, seed, filename)


class Kaggle(BinaryClassifierEval):
    def __init__(self, args, seed = 1111, filename = None):
        logging.debug('***** Task : ' + self.__class__.__name__ + ' *****\n\n')
        super(self.__class__, self).__init__(args, seed, filename)


class TREC(BinaryClassifierEval):
    def __init__(self, args, seed = 1111, filename = None):
        logging.debug('***** Task : ' + self.__class__.__name__ + ' *****\n\n')
        super(self.__class__, self).__init__(args, seed, filename)
