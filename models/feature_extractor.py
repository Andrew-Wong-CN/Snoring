from models.inception import Inception
from models.bilstm import BiLSTMBlock
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # inception block * 2 for feature extraction
        self.Inception = Inception()

        # bi-lstm block * 1 for temporal aggregation
        self.BiLSTM = BiLSTMBlock(input_size=10, hidden_size=5)

        # 1d convolutional layer, feature fusion, 10 features to 1
        self.Conv1D = nn.Conv1d(in_channels=469, out_channels=469, kernel_size=10)

    def forward(self, input_):
        output = self.Inception(input_)
        output = self.BiLSTM(output)
        output = self.Conv1D(output)
        return output
