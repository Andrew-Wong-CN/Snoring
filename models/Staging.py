# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月21日
"""
import torch

from models.inception import Inception
from models.blstm import BlstmBlock
from models.classification import Classification
import torch.nn as nn

class Staging(nn.Module):
    def __init__(self):
        super(Staging, self).__init__()
        # 用两个batchnorm分别对幅度谱，相位谱进行归一化处理, ba
        self.BatchNorm_mag = nn.BatchNorm2d(2)
        self.BatchNorm_pha = nn.BatchNorm2d(2)
        # inception block * 6 提取特征
        self.Inception_block = Inception()
        # blstm block * 2 时域融合
        self.BlstmBlock = BlstmBlock(input_size=40)
        # fully connect *2 + softmax 定位头
        self.Classification = Classification(in_features=40, out_features=10, mid_features=20)

    def forward(self,input):
        """

        :param input: 输入数据的应该包含四个维度，第一个维度应该为声道数*2，
        :return:
        """
        if input.dim() != 4:
            raise (ValueError("expected 4D input, got {}D".format(input.dim())))

        if input.shape[1] % 2 != 0:
            raise (ValueError("expected input 2nd dimension is even number, got {}".format(input.shape[0])))
        mid = input.shape[1] // 2
        mag = self.BatchNorm_mag(input[:, :mid, :, :])
        pha = self.BatchNorm_pha(input[:, mid:, :, :])
        input = torch.cat((mag, pha), dim=1)

        output = self.Inception_block(input)

        output = self.BlstmBlock(output)
        output, _ = self.Classification(output)
        return output

# if __name__ == '__main__':
#     # import torch
#     # s = torch.randn([128,4,470,80])
#     # model = Staging()
#     # predict = model(s)
