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
from torchsummary import summary

class Staging(nn.Module):
    def __init__(self):
        super(Staging, self).__init__()
        # 用两个batchnorm分别对幅度谱，相位谱进行归一化处理, ba
        self.BatchNorm_mag = nn.BatchNorm2d(2) # num features, represent C in (N, C, H, W)
        self.BatchNorm_pha = nn.BatchNorm2d(2)
        # inception block * 6 提取特征
        self.Inception_block = Inception()
        # blstm block * 2 时域融合
        self.BlstmBlock = BlstmBlock(input_size=10, hidden_size=5) # note: here only use one layer bi-lstm
        # fully connect *2 + softmax 定位头
        self.Classification = Classification(in_features=4690, out_features=5, mid_features=469)

    def forward(self, input_):
        """

        :param input_: 输入数据的应该包含四个维度，第一个维度应该为声道数*2，
        :return:
        """
        # if input.dim() != 4:
        #     raise (ValueError("expected 4D input, got {}D".format(input.dim())))
        #
        # if input.shape[1] % 2 != 0:
        #     raise (ValueError("expected input 2nd dimension is even number, got {}".format(input.shape[0])))
        # mid = input.shape[1] // 2 # '//' represents floor division. This // operator divides the first number by the second number and rounds the result down to the nearest integer (or whole number).
        # mag = self.BatchNorm_mag(input[:, :mid, :, :])
        # pha = self.BatchNorm_pha(input[:, mid:, :, :])
        # input = torch.cat((mag, pha), dim=1)

        output = self.Inception_block(input_)
        output = self.BlstmBlock(output)
        output1, output2 = self.Classification(output)

        return output1, output2

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model1 = Inception()
    # model1 = model1.to(device)
    # summary(model1, (2, 469, 20), 16)
    # model2 = BlstmBlock(input_size=10, hidden_size=5)
    # model2 = model2.to(device)
    # summary(model2, (16, 469, 10))
    model3 = Classification(in_features=4690, out_features=5, mid_features=469)
    model3 = model3.to(device)
    pytorch_total_params = sum(p.numel() for p in model3.parameters())
    print(pytorch_total_params)