from models.inception import Inception
from models.bilstm import BiLSTMBlock
from models.classification import Classification
import torch.nn as nn

class StagingNet(nn.Module):
    def __init__(self):
        super(StagingNet, self).__init__()

        # inception block * 2 feature extraction
        self.Inception = Inception()

        # bi-lstm block * 1 temporal aggregation
        self.BiLSTM = BiLSTMBlock(input_size=10, hidden_size=5)

        # fully connect *2 classification
        self.Classification = Classification(in_features=4690, out_features=5, mid_features=469)

    def forward(self, input_):
        output = self.Inception(input_)
        output = self.BiLSTM(output)
        output1, output2 = self.Classification(output)
        return output1, output2

