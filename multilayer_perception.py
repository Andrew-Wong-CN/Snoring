# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2023年02月26日
"""
import os.path

import librosa
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np

BATCH_SIZE = 256
TRAIN_SIZE = 8
TEST_SIZE = 2
label_file_path = "label file path"
spliced_path = "the new spliced dataset path"


class SplicedDataset(Dataset):
    def __init__(self, label_file, dataset_path):
        self.label = pd.read_csv(label_file)
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = np.load(self.dataset_path)
        mel_frame = mel[idx]
        label = self.labels.iloc[idx, 1]
        return mel_frame, label


def main():
    # load data
    spliced_dataset = SplicedDataset(label_file=label_file_path, dataset_path=spliced_path)
    train_spliced_dataset, test_spliced_dataset = torch.utils.data.random_split(spliced_dataset,
                                                                                [TRAIN_SIZE, TEST_SIZE])
    mpssc_train_loader = DataLoader(dataset=train_spliced_dataset, batch_size=BATCH_SIZE, shuffle=True)
    mpssc_test_loader = DataLoader(dataset=test_spliced_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_audio, train_labels = next(iter(mpssc_train_loader))
    test_audio, test_labels = next(iter(mpssc_test_loader))
    # train and test the mlp
    clf = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4,
                        hidden_layer_sizes=(100, 100), random_state=1, verbose=10, learning_rate_init=.1).fit(
        train_audio,
        train_labels)
    print(clf.score(test_audio, test_labels))
    print(clf.n_iter_)
    print(clf.loss_)
    print(clf.out_activation_)

    # predict on the original audio and store it
    pre = clf.predict()
    np.savetxt('snore_detection.csv', pre, delimiter=',')


if __name__ == '__main__':
    main()
