# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月14日
"""
import torch
import torch.nn as nn


class BlstmBlock(nn.Module):
    def __init__(self, input_size, hidden_size=20):
        """

        :param input_size: number of input features
        :param hidden_size: number of hidden layer
        """
        self.hidden_size = hidden_size
        super(BlstmBlock, self).__init__()
        self.random = False
        self.Bilsm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=1, bidirectional=True,batch_first=True)

    def forward(self, input):
        batch = input.shape[1]
        time_resolution = input.shape[0]
        if not self.random:
            output, (hn, cn) = self.Bilsm(input)
            return output
        h0 = torch.randn(time_resolution, batch, self.hidden_size)
        c0 = torch.randn(time_resolution, batch, self.hidden_size)
        output, (hn, cn) = self.Bilsm(input, (h0, c0))
        return output


class Blstm(nn.Module):

    def __init__(self, frequency=40, hidden_size=64, BR=True):
        super(BlstmBlock, self).__init__()
        self.Blstms1 = BlstmBlock(input_size=frequency, hidden_size=hidden_size)
        self.Blstms2 = BlstmBlock(input_size=2 * hidden_size, hidden_size=int(hidden_size / 2))
        self.BatchNorm = nn.BatchNorm1d(num_features=hidden_size)
        self.relu = nn.PReLU()
        self.BR = BR

    def forward(self, input):
        input = torch.transpose(input, 0, 1)
        output = self.Blstms1(input)
        output = self.Blstms2(output)
        output = torch.transpose(output, 0, 1)
        if self.BR:
            output = torch.transpose(output, 1, 2)
            output = self.relu(self.BatchNorm(output))
            output = torch.transpose(output, 1, 2)
        return output


if __name__ == '__main__':
    from torchinfo import summary

    model = Blstm(frequency=128)

    summary(model, input_size=(356, 128), batch_size=2, device='cpu')
