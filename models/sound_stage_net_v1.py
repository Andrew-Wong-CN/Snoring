from models.inception import Inception
from models.bilstm import BiLSTMBlock
from models.classifier import Classifier
import torch.nn as nn

class SoundStageNetV1(nn.Module):
    def __init__(self):
        super(SoundStageNetV1, self).__init__()

        # inception block * 2 for feature extraction
        self.Inception = Inception()

        # bi-lstm block * 1 for temporal aggregation
        self.BiLSTM = BiLSTMBlock(input_size=10, hidden_size=5)

        # fully connect layer * 2 for classification
        self.Classifier = Classifier(in_features=4690, out_features=5, mid_features=469)

    def forward(self, input_):
        output = self.Inception(input_)
        output = self.BiLSTM(output)
        output = output.reshape(output.shape[0], output.shape[1] * output.shape[2])
        output1, output2 = self.Classifier(output)
        return output1, output2

