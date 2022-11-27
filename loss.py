# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月21日
"""
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"


def loss_cross(output, label):
    '''

    :param output: [B,T,5]
    :param label: [B]
    :return:
    '''
    # category = torch.arange(0, 5, 1)
    # category = category.to(device)
    # one-hot encoding label
    # pred_1 = torch.sum(torch.mul(category,output),dim=-1) # ??? 算加权平均，平均是哪个类，如果哪个类的概率高则偏向哪个类
    # pred_1 = pred_1.to(device)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss = loss_fn(pred_1,label)

    target = one_hot(label, num_classes=5)
    target = target.to(device)
    log_softmax = nn.LogSoftmax(dim=1)
    pred = log_softmax(output)
    loss = - pred * target
    loss = torch.mean(torch.sum(loss, dim=1))

    return loss

# convert labels to one_hot
def one_hot(labels, num_classes):
    rows = labels.shape[0]
    output = torch.zeros((rows, num_classes))
    for row in range(rows):
        label = labels[row]
        output[row][label] = 1
    return output
