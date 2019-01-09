#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/1
from __future__ import absolute_import
import torch
import time
import datetime
import numpy as np
from models.model1 import BaseLineRNN
from datasets.binary import *
from utils.util import logger, accuracy
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

USE_CUDA = torch.cuda.is_available()


def train(args):
    train_dataloader, test_dataloader, model = init_from_scrach(args)
    best_acc = 0.0
    best_epoch = 0
    iter = 0
    logger('Begin training...')

    # FIXME : could modified to fit your model and algo
    if args.log_dir:
        logger_path = '../logs/log-av%s-%s-model%s-emb%d-id%s' % (
            args.activation, args.dataset, model.__class__.__name__, args.embedding_dim, str(datetime.datetime.now()))
        logger('Save log to %s' % logger_path)
        writer = SummaryWriter(log_dir = logger_path)
    for i in range(args.num_epoches):
        loss_sum = 0
        samples_num = 0
        matrics_value_sum = {}
        for j, data in enumerate(train_dataloader):
            iter += 1  # recorded for tensorboard

            # forward and loss
            model.optimizer.zero_grad()
            model.zero_grad()
            # TODO: you can modified here
            out, feature = model(*data)  # model should return the output not only predict result.
            loss = model.loss(out, data[-1])

            # backward
            loss.backward()

            # grad clip if args.grad_clipping != 0
            if args.grad_clipping != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)

            # optimize
            model.optimizer.step()

            # record
            loss_sum += loss.item()
            samples_num += len(data[0])
            matrics_value = metric(out = out.data.cpu().numpy(), label = data[-1])
            for k, v in matrics_value:
                if k in matrics_value_sum:
                    matrics_value_sum[k] = matrics_value_sum[k] + v
                else:
                    matrics_value_sum.setdefault(v)

            if (j + 1) % args.print_every_n == 0:
                info_add = ''
                for k, v in matrics_value_sum.items():
                    info_add += '| {} : {}'.format(k, v / samples_num)
                info = 'train: Epoch = %d | iter = %d/%d | loss sum = %.2f ' % (i, j, len(train_dataloader), loss_sum * 1.0 / j) + info_add
                logging.info(info)

                # for tensorboard
                if args.log_dir:
                    writer.add_scalar('loss', loss_sum / (j + 1), iter)
                    for k, v in matrics_value_sum.items():
                        writer.add_scalar(k, v / samples_num, iter)

                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(name, param.clone().cpu().data.numpy(), j)
                            writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), j)
        # Test
        logging.info("Testing...... | Model : {0} | Task : {1}".format(model.__class__.__name__, train_dataloader.dataset.__class__.__name__))
        testacc, _ = evaluation(args, model, test_dataloader)
        if best_acc < testacc:
            model.save(datetime = datetime.datetime.now())
        best_acc, best_epoch = testacc, i if best_acc < testacc else best_acc, best_epoch
        logging.error('Test result acc1: %.4f | best acc: %.4f | best epoch : %d' % (testacc, best_acc, best_epoch))


def metric(**args):
    """
    Re-implement your metric here. we use this metric to save best weight and show. if multi-metric return. use the first as default.
    :param args:
    :return:  Note: you must return a dictionary, otherwise we will not show them up.
    """
    return {'acc': accuracy(args['out'], args['label'])}


def evaluation(args, model, data_loader):
    model.eval()
    samples_num = 0
    acc_sum = 0.0

    pred = list()

    for j, a_data in enumerate(data_loader):
        out, _ = model(*a_data)
        pred.extend(out.data.cpu().numpy().max(-1)[1].tolist())
        samples_num += len(a_data[0])
        matrics = metric(out = out.data.cpu().numpy(), label = a_data[-1])
        acc_sum += matrics[matrics.keys()[0]]
    model.train()
    acc, pred = acc_sum / samples_num, pred
    save_pred(args, pred, data_loader.dataset)
    return acc, pred


def save_pred(args, pred, data):
    """
    if you want to save the prediction, just implement it
    :param args:
    :param pred:
    :param data:
    :return:
    """
    pass


def init_from_scrach(args):
    """
    init the model and load the datasets
    :param args:
    :return:
    """
    logger('No trained model provided. init model from scratch...')

    logger('Load the train dataset...')
    if args.dataset.lower() == 'cr':
        train_dataset = CR(args, filename = args.train_file)
        valid_dataset = CR(args, filename = args.valid_file)
    else:
        raise ("No dataset named {}, please check".format(args.dataset.lower()))

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False,
                                  collate_fn = train_dataset.__class__.batchfy_fn, pin_memory = True, drop_last = False)
    logger('Train data max length : %d' % train_dataset.max_len)

    logger('Load the test dataset...')
    valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = args.batch_size, shuffle = False,
                                  collate_fn = valid_dataset.__class__.batchfy_fn, pin_memory = True, drop_last = False)
    logger('Valid data max length : %d' % valid_dataset.max_len)

    logger('Initiating the model...')
    model = BaseLineRNN(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
                        rnn_layers = args.num_layers,
                        bidirection = args.bidirectional, num_class = train_dataset.num_class)

    if USE_CUDA:
        model.cuda()
    model.init_optimizer()
    logger('Model {} initiate over...'.format(model.__class__.__name__))
    logger(model)
    return train_dataloader, valid_dataloader, model
