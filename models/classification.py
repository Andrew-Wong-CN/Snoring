import torch.nn as nn


class FC(nn.Module):
    def __init__(self, in_features, out_features):
        """
        :param in_features: number of input channels
        :param out_features: number of output channels
        """
        super(FC, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.BN = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.linear(x)
        output = self.BN(output)
        output = self.relu(output)
        return output


class Classification(nn.Module):
    def __init__(self, in_features, out_features, mid_features):
        super(Classification, self).__init__()
        self.Flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.FClassify1 = FC(in_features=in_features, out_features=mid_features)
        self.FClassify2 = FC(in_features=mid_features, out_features=out_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_):
        output = input_.reshape(input_.shape[0], input_.shape[1] * input_.shape[2])
        output = self.FClassify1(output)
        output = self.FClassify2(output) # size: (16 * 936 B * T, 5)
        output1 = output # output1 without softmax, used for loss;
        output2 = self.softmax(output) # output2 with softmax, used for test
        return output1, output2
