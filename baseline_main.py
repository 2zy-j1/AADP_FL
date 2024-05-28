#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


# from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar


if __name__ == '__main__':
    args = args_parser() #解析命令行参数并将其保存在args对象中。
    if args.gpu: #如果指定了GPU编号。
        torch.cuda.set_device(args.gpu)#将当前设备设置为指定的GPU
    device = 'cuda' if args.gpu else 'cpu'#根据GPU是否可用选择设备类型

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)#获取训练和测试数据集

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape#获取输入图像的形状
        len_in = 1#初始化输入长度为1
        for x in img_size:#对于输入形状的每个维度
            len_in *= x#计算输入的总长度
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)#创建一个MLP模型对象
    else:#如果模型类型未识别
        exit('Error: unrecognized model')#输出错误信息并退出程序

    # Set the model to train and send it to device.
    global_model.to(device)#将模型移动到指定的设备上
    global_model.train()#设置模型为训练模式
    print(global_model)#打印模型结构

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)#创建训练数据集的数据加载器。
    criterion = torch.nn.NLLLoss().to(device)#创建损失函数对象
    epoch_loss = []#创建一个空列表，用于保存每个epoch的平均损失

    for epoch in range(args.epochs):#对于每个epoch，显示进度条。
        batch_loss = []#创建一个空列表，用于保存每个批次的损失

        for batch_idx, (images, labels) in enumerate(trainloader):#对于每个批次，使用enumerate获取批次索引和数据。
            images, labels = images.to(device), labels.to(device)#将数据移动到指定的设备上。

            optimizer.zero_grad()#清零优化器的梯度
            outputs = global_model(images)#通过模型进行推断得到输出
            loss = criterion(outputs, labels)#计算损失
            loss.backward()#进行反向传播计算梯度
            optimizer.step()#更新模型参数

            if batch_idx % 50 == 0:#如果批次索引是50的倍数
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(#打印训练信息，包括当前epoch、当前批次的进度、损失值。
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())#将当前批次的损失值添加到列表中。

        loss_avg = sum(batch_loss)/len(batch_loss)#计算当前epoch的平均损失值
        print('\nTrain loss:', loss_avg)#打印当前epoch的平均损失值
        epoch_loss.append(loss_avg)#将当前epoch的平均损失值添加到列表中。

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)#绘制训练损失随时间的变化曲线。
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)#使用测试数据集进行模型推断，并获取测试准确率和损失。
    print('Test on', len(test_dataset), 'samples')#打印测试样本数
    print("Test Accuracy: {:.2f}%".format(100*test_acc))#打印测试准确率
