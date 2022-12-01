from models.feature_extractor import FeatureExtractor
from models.classifier import Classifier
import torch.nn as nn

class SoundStageNetV2(nn.Module):
    def __init__(self):
        super(SoundStageNetV2, self).__init__()

        # feature extractor
        self.FeatureExtractor = FeatureExtractor()

        # fully connect layer * 2 for classification
        self.Classifier = Classifier(in_features=469, out_features=5, mid_features=50)

    def forward(self, input_):
        output = self.FeatureExtractor(input_)
        output = output.reshape((output.shape[0], output.shape[1]))
        output1, output2 = self.Classifier(output)
        return output1, output2
