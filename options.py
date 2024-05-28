#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    #训练轮数5
    parser.add_argument('--epochs', type=int, default=30,
                        help="number of rounds of training")
    # 用户数量30
    parser.add_argument('--num_users', type=int, default=30,
                        help="number of users: K")
    #客户端比例，用于联邦更新的用户比例。默认为0.1
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    # 每个用户的本地训练轮数4
    parser.add_argument('--local_ep', type=int, default=4,
                        help="the number of local epochs: E")
    # 每个用户的本地更新的批量大小。80
    parser.add_argument('--local_bs', type=int, default=80,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments模型名称，默认值: 'mlp'。选项: 'mlp'、'cnn'
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    # 卷积层中不同大小卷积核的数量
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    # 卷积核的尺寸
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    # 图像通道数
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    # 隐私预算
    parser.add_argument('--epsilon', type=float, default=3, help='epsilon')
    #百分位数
    parser.add_argument('--percentile', type=int, default=0, help='percentile')
    # 误差数
    parser.add_argument('--h', type=int, default=0.1, help='h')
    # other arguments数据集名称，默认值: 'mnist',选项: 'mnist'、'fmnist'、'cifar'
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    # 类别数
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    # GPU使用情况
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    # 优化器类型
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    # 用户数据的分布。默认设置为 IID。设置为0表示非IID。
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    #用于指定是否使用不平等的数据划分。默认值为0，表示使用平等的数据划分。
    # 在非IID设置中使用。平均分配数据给用户或不平均分配的选项。默认设置为0以进行平均分配。设置为1以进行不平均分配。
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    # 详细的日志输出，默认激活，设置为0以停用
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    # 随机种子，默认设置为1
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
