import torch.nn as nn


class FC(nn.Module):
    def __init__(self, in_features, out_features):
        """
        :param in_features: number of input channels
        :param out_features: number of output channels
        """
        super(FC, self).__init__()
        self.Linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.BN = nn.BatchNorm1d(out_features)
        self.ReLU = nn.ReLU()

    def forward(self, input_):
        output = self.Linear(input_)
        output = self.BN(output)
        output = self.ReLU(output)
        return output


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, mid_features):
        super(Classifier, self).__init__()
        self.FClassify1 = FC(in_features=in_features, out_features=mid_features)
        self.FClassify2 = FC(in_features=mid_features, out_features=out_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_):
        output = self.FClassify1(input_)
        output = self.FClassify2(output) # size: (16 * 936 B * T, 5)
        output1 = output # output1 without softmax, used for loss;
        output2 = self.softmax(output) # output2 with softmax, used for test
        return output1, output2
