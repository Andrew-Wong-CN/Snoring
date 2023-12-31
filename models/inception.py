import torch
import torch.nn as nn
import torch.nn.functional as func


class Conv2D(nn.Module):
    def __init__(self, input_size, out_channels,kernel_size, stride, padding):
        """
        :param input_size: the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size: the size of kernel
        :param stride: stride
        :param padding: number of padding
        """
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(input_size, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN = nn.BatchNorm2d(out_channels)
        self.Relu = nn.PReLU(out_channels)

    def forward(self, x):
        output = self.conv(x)
        output = self.BN(output)
        output = self.Relu(output)
        return output


class InceptionBlock(nn.Module):
    def __init__(self, depth_dim, input_size, config, num='in2'):
        """
        :param depth_dim: determine which dimension is the channel
        :param input_size: number of input channels
        :param config: number of filters
        """
        super().__init__()
        self.depth_dim = depth_dim
        self.conv1 = Conv2D(input_size, out_channels=config[0][0], kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv3_1 = Conv2D(input_size, out_channels=config[1][0], kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv3_3 = Conv2D(config[1][0], config[1][1], kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5_1 = Conv2D(input_size, out_channels=config[2][0], kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv5_5 = Conv2D(config[2][0], config[2][1], kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1)
        self.conv_max_1 = Conv2D(input_size, out_channels=config[3][1], kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.BatchNorm = nn.BatchNorm2d(num_features=config[0][0] + config[1][1] + config[3][1] + config[2][1])
        self.num = num

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv3_1(x)
        output2 = self.conv3_3(output2)
        output3 = self.conv5_1(x)
        output3 = self.conv5_5(output3)
        output4 = self.max_pool_1(x)
        output4 = self.conv_max_1(output4)
        return func.relu(self.BatchNorm(torch.cat([output1, output2, output3, output4], dim=self.depth_dim)))

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.inception_1 = InceptionBlock(depth_dim=1, input_size=2, config=[[2], [4, 8], [1, 2], [3, 2]], num='in1')
        self.inception_2 = InceptionBlock(1, 14, [[2], [2, 4], [1, 2], [3, 2]])
        # self.inception_3 = InceptionBlock(1, 10, [[2], [2, 4], [1, 2], [3, 2]])
        # self.inception_4 = InceptionBlock(1, 10, [[2], [2, 4], [1, 2], [3, 2]])
        # self.inception_5 = InceptionBlock(1, 10, [[2], [2, 4], [1, 2], [3, 2]])
        # self.inception_6 = InceptionBlock(1, 10, [[2], [2, 4], [1, 2], [3, 2]])
        self.max_pool_12 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.conv256 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=(1, 1))

    def forward(self, input_):
        output = input_
        output = self.inception_1(output)
        output = self.inception_2(output)
        # output = self.inception_3(output)
        # output = self.inception_4(output)
        # output = self.inception_5(output)
        # output = self.inception_6(output)
        output = self.max_pool_12(output)
        output = self.conv256(output)
        output = torch.squeeze(output, dim=1)
        return output
