from models.feature_extractor import FeatureExtractor
from models.classifier import Classifier
from models.bilstm import BiLSTMBlock
import torch.nn as nn

class PreTrainingNet(nn.Module):
    def __init__(self):
        super(PreTrainingNet, self).__init__()

        # feature extractor
        self.FeatureExtractor = FeatureExtractor()

        # fully connect layer * 2 for classification
        self.Classifier = Classifier(in_features=469, out_features=5, mid_features=50)

    def forward(self, input_):
        output = self.FeatureExtractor(input_)
        output = output.reshape((output.shape[0], output.shape[1]))
        output1, output2 = self.Classifier(output)
        return output1, output2

class SoundStageNetV2(nn.Module):
    def __init__(self):
        super(SoundStageNetV2, self).__init__()

        # feature extractor
        self.FeatureExtractor = FeatureExtractor()

        # temporal aggregation
        self.BiLSTM = BiLSTMBlock(input_size=469, hidden_size=200, random=False)

        # classifier
        self.Classifier = Classifier(in_features=400, out_features=5, mid_features=40)

    def forward(self, input_):
        output = self.FeatureExtractor(input_)
        output = output.reshape((output.shape[0], output.shape[1]))
        output = self.BiLSTM(output)
        output1, output2 = self.Classifier(output)
        return output1, output2

class SoundStageNetV2Test(nn.Module):
    def __init__(self):
        super(SoundStageNetV2Test, self).__init__()

        # feature extractor
        self.FeatureExtractor = FeatureExtractor()

        # temporal aggregation
        self.BiLSTM = BiLSTMBlock(input_size=469, hidden_size=200, random=False)

    def forward(self, input_):
        output = self.FeatureExtractor(input_)
        output = output.reshape((output.shape[0], output.shape[1]))
        output = self.BiLSTM(output)
        return output