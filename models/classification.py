# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月20日
"""
import torch
import torch.nn as nn


class FBR(nn.Module):
    def __init__(self, in_features, out_features):
        """
        :param in_features: number of input channels
        :param out_features: number of output channels
        """
        super(FBR, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.BN = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.linear(x)
        output = self.BN(output)
        output = self.relu(output)
        return output


class Classification(nn.Module):
    def __init__(self, in_features=40, out_features=5, mid_features=10):
        super(Classification, self).__init__()
        # print('###init classification###')
        self.FClassify1 = FBR(in_features=in_features, out_features=mid_features)
        self.FClassify2 = FBR(in_features=mid_features, out_features=out_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        input0 = input.shape[0]
        input1 = input.shape[1]
        input = input.reshape(input0 * input1, -1)
        output = self.FClassify1(input)
        output = self.FClassify2(output)
        output2 = output
        output = self.softmax(output)
        output = output.reshape(input0, input1, -1)
        return output, output2


if __name__ == '__main__':
    x = torch.randn((2))
    print(x)
    import math

    y = torch.randn((2))
    x = torch.clamp(x, min=-math.pi, max=math.pi)
    y = torch.clamp(y, min=-math.pi, max=math.pi)
    # from sklearn.metrics.pairwise import haversine_distances
    #
    # result = haversine_distances([x, y])
    print(1 / torch.sqrt(1 - torch.cos(torch.tensor(0.0003))))
