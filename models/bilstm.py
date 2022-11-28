import torch
import torch.nn as nn

class BiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        super(BiLSTMBlock, self).__init__()
        self.random = False
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, input_):
        batch = input_.shape[1]
        time_resolution = input_.shape[0]
        if not self.random:
            output, (_, _) = self.bi_lstm(input_)
            return output
        h0 = torch.randn(time_resolution, batch, self.hidden_size)
        c0 = torch.randn(time_resolution, batch, self.hidden_size)
        output, (_, _) = self.bi_lstm(input_, (h0, c0))
        return output


class BiLSTM(nn.Module):
    def __init__(self, frequency=40, hidden_size=20):
        super(BiLSTM, self).__init__()
        self.BiLSTM1 = BiLSTMBlock(input_size=frequency, hidden_size=hidden_size)
        self.BiLSTM2 = BiLSTMBlock(input_size=2 * hidden_size, hidden_size=int(hidden_size / 2))
        self.BatchNorm = nn.BatchNorm1d(num_features=hidden_size)
        self.relu = nn.PReLU()

    def forward(self, input_):
        input_ = torch.transpose(input_, 0, 1)
        output = self.BiLSTM1(input_)
        output = self.BiLSTM2(output)
        output = torch.transpose(output, 0, 1)
        if self.BR:
            output = torch.transpose(output, 1, 2)
            output = self.relu(self.BatchNorm(output))
            output = torch.transpose(output, 1, 2)
        return output

