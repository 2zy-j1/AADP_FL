#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
import random

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    # 初始化函数，接收dataset和idxs作为参数
    def __init__(self, dataset, idxs):
        self.dataset = dataset#将传入的dataset赋值给类属性dataset
        self.idxs = [int(i) for i in idxs]#将传入的idxs转换为整型并赋值给类属性idxs。

    def __len__(self):#返回数据集的长度。
        return len(self.idxs)

    # 根据索引item从数据集中获取样本和标签
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]#根据索引获取原始数据集中的样本和标签。
        # return torch.tensor(image), torch.tensor(label)#将样本和标签转换为张量格式并返回。
        return image, label # 将样本和标签转换为张量格式并返回。


class LocalUpdate(object):
    #初始化函数，接收args、dataset、idxs和logger作为参数。

    def __init__(self, args, dataset, idxs, logger):
        self.args = args#将传入的args赋值给类属性args
        self.logger = logger#将传入的logger赋值给类属性logger
        #调用train_val_test方法，将返回的训练集、验证集和测试集加载器赋值给类属性trainloader、validloader和testloader。
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        # 根据args中的gpu属性判断设备类型，并将结果赋值给类属性device。
        self.device = 'cuda' if args.gpu else 'cpu'
        # 创建一个负对数似然损失函数对象，并将其移动到指定设备上。
        self.criterion = nn.NLLLoss().to(self.device)
    # 定义了一个用于划分训练集、验证集和测试集的方法train_val_test。
    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        # split indexes for train, validation, and test (80, 10, 10)
        # 将idxs按照80%的比例划分为训练集的索引。
        idxs_train = idxs[:int(0.8*len(idxs))]
        # 将idxs按照80%-90%的比例划分为验证集的索引。
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # 将idxs按照90%以上的比例划分为测试集的索引。
        idxs_test = idxs[int(0.9*len(idxs)):]
        # 创建训练集数据加载器，使用DatasetSplit类和划分好的训练集索引，并指定批量大小和是否打乱顺序。
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        # 创建验证集数据加载器，使用DatasetSplit类和划分好的验证集索引，并指定批量大小和是否打乱顺序。
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        # 创建测试集数据加载器，使用DatasetSplit类和划分好的测试集索引，并指定批量大小和是否打乱顺序。
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        # 返回训练集、验证集和测试集的数据加载器。
        return trainloader, validloader, testloader

    # 定义了一个用于更新模型权重的方法update_weights。 设置模型为训练模式、定义优化器。 计算损失并更新模型参数。
    def update_weights(self, model, global_round):
        # Set mode to train model

        model.train()
        epoch_loss = []
        batch_history = {}
        for i in range(len(self.trainloader)):
            batch_history[i] = []
            # Set optimizer for the local updates
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                            momentum=0.5)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                             weight_decay=1e-4)
            grad_history = {}
            # c = 0

            for iter in range(1,self.args.local_ep+1):
                # c = int(max(1, iter // 2))  # 使用整除而非浮点数除法
                c=iter
                batch_loss = []

                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    # 清除模型参数的梯度
                    model.zero_grad()

                    # 前向传播
                    log_probs = model(images)

                    # 计算损失
                    loss = self.criterion(log_probs, labels)

                    # 反向传播
                    loss.backward()

                    # 裁剪梯度
                    clipped_grads = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if name not in grad_history:
                                grad_history[name] = []
                            grad_norm = torch.norm(param.grad, p=2)
                            grad_history[name].append(grad_norm.item())
                            if len(grad_history[name]) > c:
                                grad_history[name].pop(0)
                            recent_l2_norms = grad_history[name]
                            threshold = sum(recent_l2_norms) / len(recent_l2_norms)
                            if grad_norm > threshold:
                                clipped_grad = (param.grad / grad_norm) * threshold
                            else:
                                clipped_grad = param.grad
                            clipped_grads.append(clipped_grad)
                            # 注意：这里不实际更新param.grad，因为optimizer.step()会处理

                    # 更新优化器
                    optimizer.step()
                    optimizer.zero_grad()  # 重置梯度为0，以便下一次迭代
                    # 记录损失
                    batch_loss.append(loss.item())
                    self.logger.add_scalar('loss', loss.item())

                    # 如果这是最后一个批次，并且需要添加噪声
                    # if batch_idx == len(self.trainloader):
                    if iter == self.args.local_ep+1:
                        for name, param in model.named_parameters():
                                sc=0
                                if param.grad is not None:
                                    if name not in grad_history:
                                        grad_history[name] = []
                                    grad_norm = torch.norm(param.grad, p=2)
                                    grad_history[name].append(grad_norm.item())
                                    if len(grad_history[name]) > c:
                                        grad_history[name].pop(0)
                                    recent_l2_norms = grad_history[name]
                                    threshold = sum(recent_l2_norms) / len(recent_l2_norms)
                                    if grad_norm > threshold:
                                        clipped_grad = (param.grad / grad_norm) * threshold
                                    else:
                                        clipped_grad = param.grad
                                    sc=sc+clipped_grad
                        explions=sc/sum(grad_history[name])
                        f=random.uniform(-0.5, 0.5)
                        μ=0
                        if f<0:
                            noise_scale = μ+(2*self.args.lr*clipped_grad*math.log(1+2*f))/explions
                        else:
                            noise_scale = μ + (2 * self.args.lr * clipped_grad * math.log(1 - 2 * f)) / explions

                            noise = torch.randn_like(param) * noise_scale
                            param.data.add_(noise)
        return model.state_dict(), sum(batch_loss) / len(batch_loss)

                    #    # 更新本地参数
                    # for param, clipped_grad in zip(model.parameters(), clipped_gradients):
                    #     if param.grad is not None:
                    #         param.data=clipped_grad
                    # optimizer.step()
                    # # print("loss", loss.item())
                    # self.logger.add_scalar('loss', loss.item())
                    # batch_loss.append(loss.item())
                    # epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    # if iter == self.args.local_ep:
                    # # 当前层比上全部层裁剪后的梯度范数
                    #   sum_clipped_grads = clipped_grads.sum()
                    #   sum_clipped_gradients = clipped_gradients.sum()
                    #   explions = sum_clipped_grads / sum_clipped_gradients
                    #   μ = 0
                    #   # 将每层梯度裁剪并加噪声
                    #   for grad, threshold in zip(layer_gradients, threshold):
                    #      clipped_grad = torch.clamp(grad, max=threshold)
                    #      # f=random.uniform(-0.5, 0.5)
                    #      f=0.1
                    #
                    #      if f<0:
                    #          noisy_grad = μ+(2*self.args.lr*clipped_grad*math.log(1+2*f))/explions
                    #      else:
                    #          noisy_grad = μ + (2 * self.args.lr * clipped_grad * math.log(1 - 2 * f)) / explions
                    #      print('noisy', noisy_grad)
                    #      param.data.add_(-self.args.lr * noisy_grad)



    # 定义了一个用于推断模型的方法inference
    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        # 将模型设置为评估模式
        model.eval()
        # 以0.0初始化损失、总数和正确预测数。  迭代测试集数据进行推断，计算损失并统计正确预测数。
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to('cpu')

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        # 返回推断的准确率和损失
        return accuracy, loss

# 定义了一个用于在测试集上进行推断的方法test_inference。
def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    # 将模型设置为评估模式
    model.eval()
    # 0.0初始化损失、总数和正确预测数。
    # 通过DataLoader对测试集进行加载。 86-91. 迭代测试集数据进行推断，计算损失并统计正确预测数。
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    # 返回推断的准确率和损失
    return accuracy, loss
