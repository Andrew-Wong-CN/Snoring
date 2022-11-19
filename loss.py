# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月21日
"""
import torch.nn


def loss_cross(output, label):
    '''

    :param output: [B,T,5]
    :param label: [B]
    :return:
    '''
    category =torch.arange(0,5,1)
    #.todevice()
    pred_1 =torch.sum(torch.mul(category,output),dim=-1)


    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(pred_1,label)
    return loss
