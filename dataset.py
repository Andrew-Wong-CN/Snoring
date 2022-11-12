# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月04日
"""
import os
import librosa
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import preprocessing


class SnoringDataset(Dataset):
    def __init__(self, label_file, dataset_path):
        """

        :param label_file: label file (.csv)
        :param dataset_path: subjectyydd/snoring
        """

        self.labels = pd.read_csv(label_file)
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 按照索引获取标签及录音文件
        audio_path = os.path.join(self.dataset_path, f'{self.labels.iloc[idx, 0]}.wav')
        audio = librosa.load(audio_path, sr=32000)
        label = self.labels.iloc[idx, 1]
        # 将string类型label与数值映射
        # WK:4   N1:0    N2:1    N3:2    REM:3
        # label_encoder = preprocessing.LabelEncoder()
        # label_encoder.fit(['WK', 'N1', 'N2', 'N3', 'REM'])
        # label_transed = label_encoder.transform(label)
        # label_encoder使用注意：输入必须是一个数组，相当于批量转换label
        label_transed = stage_dict.get(label,'5')
        return audio, label_transed


stage_dict = {
    #单向map
    'N1': 0,
    'N2': 1,
    'N3': 2,
    'REM':3,
    'WK': 4,
}


if __name__ == '__main__':
    import torch

    dataset = SnoringDataset(
        label_file='D:\\Ameixa\\学习\\实验室\\Snoring Detection\\DataSet\\Subject0905\\SleepStaging.csv',
        dataset_path='D:\\Ameixa\\学习\\实验室\\Snoring Detection\\DataSet\\Subject0905\\Snoring')

    print(len(dataset))
    train_size = int(len(dataset) * 0.7)
    test_size = int(len(dataset) * 0.3)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(len(train_loader))
    print(len(test_loader))
