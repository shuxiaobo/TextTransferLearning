#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Desc: 
    @Author: shane 
    @Contact: iamshanesue@gmail.com
    @Software: PyCharm
    @Since: Python3.6
    @Date: 2019/1/9
    @All right reserved
"""  # !/usr/bin/env python3
# coding: utf-8
# File: cbow.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-21

import os
import math
import argparse
import numpy as np
import collections
import tensorflow as tf
from tensorflow.python.ops import init_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class DataLoader:
    def __init__(self):
        self.datafile = 'data/data.txt'
        self.dataset = self.load_data()

    '''加载数据集'''

    def load_data(self):
        dataset = []
        for line in open(self.datafile):
            line = line.strip().split(' ')
            dataset.append(line)
        return dataset


class CBOW:
    def __init__(self, trainfilepath = './data/data', num_sampled = 100, num_steps = 100000, window_size = 1, embedding_size = 200, batch_size = 200,
                 min_count = 5, name = '', is_ngram = 0):
        self.data_index = 0
        self.min_count = min_count  # 默认最低频次的单词
        self.batch_size = batch_size  # 每次迭代训练选取的样本数目
        self.embedding_size = embedding_size  # 生成词向量的维度
        self.window_size = window_size  # 考虑前后几个词，窗口大小
        self.num_steps = num_steps  # 定义最大迭代次数，创建并设置默认的session，开始实际训练
        self.num_sampled = num_sampled  # Number of negative examples to sample.
        self.trainfilepath = trainfilepath
        self.modelpath = os.path.join('..', 'save', name, 'cbow_wordvec.bin')
        self.dataset = DataLoader().dataset
        self.words = self.read_data(self.dataset)
        self.is_ngram = is_ngram

    # 定义读取数据的函数，并把数据转成列表
    def read_data(self, dataset):
        words = []
        for data in dataset:
            words.extend(data)
        return words

    # 创建数据集
    def build_dataset(self, words, min_count):
        # 创建词汇表，过滤低频次词语，这里使用的人是mincount>=5，其余单词认定为Unknown,编号为0,
        # 这一步在gensim提供的wordvector中，采用的是minicount的方法
        # 对原words列表中的单词使用字典中的ID进行编号，即将单词转换成整数，储存在data列表中，同时对UNK进行计数
        count = [['UNK', -1]]
        reserved_words = [item for item in collections.Counter(words).most_common() if item[1] >= min_count]
        count.extend(reserved_words)
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        count[0][1] = unk_count
        print(len(count))
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary

    # 生成训练样本，assert断言：申明其布尔值必须为真的判定，如果发生异常，就表示为假
    def generate_batch(self, batch_size, skip_window, data):
        # 该函数根据训练样本中词的顺序抽取形成训练集
        # batch_size:每个批次训练多少样本
        # skip_window:单词最远可以联系的距离（本次实验设为5，即目标单词只能和相邻的两个单词生成样本），2*skip_window>=num_skips
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        batch = np.ndarray(shape = (batch_size, span - 1), dtype = np.int32)
        labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
        buffer = collections.deque(maxlen = span)

        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_indexdata_index = (self.data_index + 1) % len(data)

        for i in range(batch_size):
            target = skip_window
            target_to_avoid = [skip_window]
            col_idx = 0
            for j in range(span):
                if j == span // 2:
                    continue
                batch[i, col_idx] = buffer[j]
                col_idx += 1
            labels[i, 0] = buffer[target]

            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)

        assert batch.shape[0] == batch_size and batch.shape[1] == span - 1

        return batch, labels

    def train_wordvec(self, vocabulary_size, batch_size, embedding_size, window_size, num_sampled, num_steps, data):
        # 定义CBOW Word2Vec模型的网络结构
        graph = tf.Graph()
        with graph.as_default():
            train_dataset = tf.placeholder(tf.int32, shape = [batch_size, 2 * window_size])
            train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            self.embedding = embeddings
            if self.is_ngram:
                self.gates = []
                for i in range(self.window_size, 0, -1):
                    self.gates.append(
                        tf.get_variable('gate_' % i, shape = [i * 2, self.embedding_size], initializer = init_ops.uniform_unit_scaling_initializer))
                self.gate_new = tf.get_variable('gate_new', shape = [self.window_size * self.embedding_size, embedding_size * self.window_size],
                                                initializer = init_ops.random_uniform_initializer)
            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
            # 与skipgram不同， cbow的输入是上下文向量的均值，因此需要做相应变换
            context_embeddings = []
            for i in range(2 * window_size):
                context_embeddings.append(tf.nn.embedding_lookup(embeddings, train_dataset[:, i]))
            avg_embed = tf.reduce_mean(tf.stack(axis = 0, values = context_embeddings), 0, keep_dims = False)
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights = softmax_weights, biases = softmax_biases, inputs = avg_embed,
                                           labels = train_labels, num_sampled = num_sampled,
                                           num_classes = vocabulary_size))
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
            normalized_embeddings = embeddings / norm

        with tf.Session(graph = graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            for step in range(num_steps):
                batch_data, batch_labels = self.generate_batch(batch_size, window_size, data)
                feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
                _, l = session.run([optimizer, loss], feed_dict = feed_dict)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
            final_embeddings = normalized_embeddings.eval()
        return final_embeddings

    # 保存embedding文件
    def save_embedding(self, final_embeddings, model_path, reverse_dictionary):
        f = open(model_path, 'w+')
        for index, item in enumerate(final_embeddings):
            f.write(reverse_dictionary[index] + '\t' + ','.join([str(vec) for vec in item]) + '\n')
        f.close()

    # 训练主函数
    def train(self):
        data, count, dictionary, reverse_dictionary = self.build_dataset(self.words, self.min_count)
        vocabulary_size = len(count)
        final_embeddings = self.train_wordvec(vocabulary_size, self.batch_size, self.embedding_size, self.window_size, self.num_sampled, self.num_steps, data)
        self.save_embedding(final_embeddings, self.modelpath, reverse_dictionary)

    def context_embedding(self, x):
        emb_list = []
        for i in range(self.window_size):
            tmp = tf.nn.embedding_lookup(self.embedding, x[:, i:-i])
            embed = tmp * tf.nn.softmax(self.gates[i], 0)
            emb_list.append(embed)
        scores = tf.split(tf.einsum("bij,jk->bik", tf.concat(emb_list, -1), self.gate_new), self.window_size, -1)
        embed = scores[-1] * emb_list[-1]
        for i in range(2, self.window_size):
            embed += emb_list[-i]
            embed *= scores[-i]
        return embed


class ContextEmbedding:
    def __init__(self, context_size = [7, 5, 3], embedding_dim = 300):
        """
        use the word context embedding
        :param context_size: list or int
        :param method: 'concat' , 'dot', 'matmul'
        """
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.gates = []
        for i in range(len(context_size)):
            self.gates.append(tf.get_variable('gate%d' % i, shape = [embedding_dim, embedding_dim], initializer = init_ops.random_uniform_initializer))
        # self.gate = tf.get_variable('gate%d' % context_size, shape = [embedding_dim, embedding_dim], initializer = init_ops.random_uniform_initializer)

        self.gate_new = tf.get_variable('gate_new', shape = [embedding_dim * len(context_size), embedding_dim * len(context_size)],
                                        initializer = init_ops.random_uniform_initializer)

    def __call__(self, x, embedding_matrix):
        """
        concat or dot or matmul the context words embedding.
        :param x:
        :param embedding_matrix:
        :return:
        """
        with tf.variable_scope('context_embedding', reuse = False) as scp:
            ngram_represent = []
            embed = tf.nn.embedding_lookup(embedding_matrix, x)
            for j in range(len(self.context_size)):
                hidden_size = tf.shape(embedding_matrix)[-1]
                middle = math.floor(self.context_size[j] / 2.0)
                embed_list = list()
                batch_size = tf.shape(x)[0]
                for i in range(-middle, 0):
                    embed_list.append(tf.nn.embedding_lookup(embedding_matrix,
                                                             tf.concat([tf.zeros(shape = [batch_size, -i], dtype = tf.int32), x[:, :i]], -1), ))
                embed_list.append(tf.nn.embedding_lookup(embedding_matrix, x))
                for i in range(1, middle + 1):
                    embed_list.append(tf.nn.embedding_lookup(embedding_matrix,
                                                             tf.concat([x[:, i:], tf.zeros(shape = [tf.shape(x)[0], i], dtype = tf.int32)], -1), ))
                stacked_emb = tf.stack(embed_list, -2)
                embed_ngram = tf.nn.softmax(tf.einsum('bijk,ks->bijs', stacked_emb, self.gates[j]), -1) * stacked_emb
                embed_ngram = tf.reduce_max(embed_ngram, -2)

                # embed = tf.reduce_max(tf.stack(embed_list, -2) * tf.nn.softmax(self.gate, 0), -2)
                ngram_represent.append(embed_ngram)
            scores = tf.split(tf.einsum("bij,jk->bik", tf.concat(ngram_represent, -1), self.gate_new), len(self.context_size), -1)

            embed_tmp = scores[0] * ngram_represent[0]
            for i in range(1, len(self.context_size)):
                embed_tmp += ngram_represent[i]
                embed_tmp *= scores[i]

        return embed + embed_tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help = 'Training file', dest = '../data/data', required = False)
    parser.add_argument('--model', help = 'Output model file', dest = 'fo', required = False)
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--cbow', help = '1 for CBOW, 0 for skip-gram', dest = 'cbow', default = 1, type = int)
    parser.add_argument('--negative', help = 'Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest = 'neg', default = 5,
                        type = int)
    parser.add_argument('--dim', help = 'Dimensionality of word embeddings', dest = 'dim', default = 100, type = int)
    parser.add_argument('--alpha', help = 'Starting alpha', dest = 'alpha', default = 0.025, type = float)
    parser.add_argument('--window', help = 'Max window length', dest = 'win', default = 5, type = int)
    parser.add_argument('--min-count', help = 'Min count for words used to learn <unk>', dest = 'min_count', default = 5, type = int)
    parser.add_argument('--epoch', help = 'Number of training epochs', dest = 'epoch', default = 100, type = int)
    parser.add_argument('--name', help = 'name of experiment', default = 'name', type = str)
    args = parser.parse_args()

    vector = CBOW(trainfilepath = args.file, num_sampled = args.negative, num_steps = args.negative, window_size = args.window, embedding_size = args.dim,
                  batch_size = args.batch_size,
                  min_count = args.min_count, name = args.name, is_ngram = args.cbow)
    vector.train()
