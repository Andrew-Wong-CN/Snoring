# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月21日
"""
import torch.nn


def crossentropyloss(output,label):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output,label)
    return loss
