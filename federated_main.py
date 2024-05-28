#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np

from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths获取项目的绝对路径，并将其赋值给变量path_project。
    path_project = os.path.abspath('..')
    # 创建一个SummaryWriter对象，用于记录实验结果和可视化，日志文件将保存在'../logs'目录下，然后将对象赋值给变量logger。
    logger = SummaryWriter('../logs')
    # 调用args_parser()函数解析命令行参数，并将返回的参数对象赋值给变量args。
    args = args_parser()
    # 调用exp_details()函数打印实验细节，传入参数对象args作为参数。
    exp_details(args)
    # 如果参数对象args中的gpu_id属性不为空，则执行下面的代码块。
    # if args.gpu_id:
    #     # 设置PyTorch使用的GPU设备为args.gpu_id指定的GPU。
    #     torch.cuda.set_device(args.gpu_id)
    # # 根据参数对象args中的gpu属性判断是否启用GPU，如果启用则将device设置为'cuda'，否则设置为'cpu'。
    # device = 'cuda' if args.gpu else 'cpu'
    device='cpu'

    # 调用get_dataset()函数获取训练集、测试集和用户组，函数接收参数对象args并返回这些数据集以及用户组。
    train_dataset, test_dataset, user_groups = get_dataset(args)

    #  判断参数对象args中的model属性是否为'cnn'，如果是则执行下面的代码块。
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            print("minist")
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            print("fminist")
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            print("cifar")
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        #  获取训练集的第一个样本的输入特征形状，并将其赋值给变量img_size。
        img_size = train_dataset[0][0].shape
        len_in = 1#初始化变量len_in为1。
        for x in img_size:#遍历img_size的每个元素，将所有元素相乘得到len_in。
            len_in *= x
            #创建一个MLP对象，传入输入维度len_in、隐藏层维度64和输出类别数args.num_classes，并将其赋值给变量global_model。
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    #如果args.model既不是'cnn'也不是'mlp'，则输出错误提示信息并终止程序的执行。
    else:
        exit('Error: unrecognized model')

    # 将全局模型global_model移动到设备device上进行计算。
    global_model.to(device)
    #将全局模型设置为训练模式
    global_model.train()
    #打印全局模型的结构。
    print(global_model)

    # 获取全局模型的权重参数，并将其赋值给变量global_weights。
    global_weights = global_model.state_dict()

    # 初始化空列表train_loss和train_accuracy，用于存储训练过程中的损失和准确率
    train_loss, train_accuracy = [], []
    #初始化空列表val_acc_list和net_list，用于存储验证集的准确率和网络模型。
    val_acc_list, net_list = [], []
    #初始化空列表cv_loss和cv_acc，用于存储交叉验证集的损失和准确率。
    cv_loss, cv_acc = [], []
    #设置打印训练过程中统计信息的间隔
    print_every = 2
    #初始化变量val_loss_pre为0，counter为0，用于记录验证集的损失和计数。
    val_loss_pre, counter = 0, 0

    # idxs_users=[26,29,20]
    # print("几个用户", len(idxs_users), idxs_users)
#循环遍历args.epochs次，显示进度条，并将每次循环的迭代次数保存在变量epoch中。
    for epoch in range(args.epochs):
        #初始化空列表local_weights和local_losses，用于存储本地更新后的权重和损失。
        local_weights, local_losses = [], []
        #打印全局训练轮数
        print(f'\n | Global Training Round : {epoch+1} |\n')
        # 将全局模型设置为训练模式
        global_model.train()
        # 计算参与本轮训练的用户数目m，取值为参与用户数的args.frac倍和1中的较大值。计算参与本轮训练的用户数目m。首先，将args.frac乘以args.num_users，然后取结果与1中的较大值，最终转换为整数。这个表达式的目的 是确保m不小于1。
        m = max(int(args.frac * args.num_users), 1)
        # 随机选择m个不重复的用户索引，范围是从0到args.num_users-1。 使用np.random.choice函数从0到args.num_users-1的范围中，随机选择m个不重复的用户索引。参数range(args.num_users)表示选择的范围是从0到args.num_users-1，参数m表示选择的数量，参数replace=False表示选择的元素不可重复。
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("用户", idxs_users)
        # 遍历每个用户的索引
        for idx in idxs_users:
            print("用户 ",idx)
            # 创建一个LocalUpdate对象，传入参数对象args、训练数据集train_dataset、当前用户的样本索引列表user_groups[idx]和记录器logger，并将这个对象赋值给变量local_model。
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            #调用update_weights()方法执行本地权重更新，传入全局模型的深拷贝copy.deepcopy(global_model)，并将更新后的权重和损失分别赋值给变量w和loss。
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            #将更新后的权重深拷贝并添加到local_weights列表中。
            local_weights.append(copy.deepcopy(w))
            # print("本地权重",local_weights)
            #将损失深拷贝并添加到local_losses列表中。
            local_losses.append(copy.deepcopy(loss))

        # 计算平均权重，调用average_weights()函数传入local_weights列表，并将计算结果赋值给global_weights。
        global_weights = average_weights(local_weights)
        # print("平均权重",global_weights)
        # 使用平均权重更新全局模型的状态字典。
        global_model.load_state_dict(global_weights)
        #计算平均损失，将local_losses列表中的损失值求和后除以列表长度。
        loss_avg = sum(local_losses) / len(local_losses)
        #将平均损失添加到train_loss列表中
        train_loss.append(loss_avg)

        # 初始化空列表list_acc和list_loss，用于存储每个用户的准确率和损失。
        list_acc, list_loss = [], []
        # 将全局模型设置为评估模式
        global_model.eval()
        #遍历每个用户，并为每个用户创建一个LocalUpdate对象
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            #调用local_model对象的inference方法，在全局模型上进行推理，得到准确率和损失，并将其存储在list_acc和list_loss列表中。
            list_acc.append(acc)
            list_loss.append(loss)
            #计算每个用户的平均准确率，并将结果存储在train_accuracy列表中
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        #如果当前轮数是指定的打印轮数（print_every）的倍数，则打印训练损失和最后一次训练的平均准确率。
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

            # Test inference after completion of training
            # 使用test_inference函数对全局模型进行测试推理，得到测试准确率和损失。
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            # 打印训练完成后的平均训练准确率和测试准确率。
            # print(f' \n Results after {args.epochs} global rounds of training:')
            print(f' \n Results after {epoch+1} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    # #将训练过程中的训练损失和准确率保存为pickle文件
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)
    #打印整个训练过程的总运行时间。
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # # 可选的代码段，用于绘制训练损失曲线和平均准确率曲线
    # # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    #
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
