import os
import librosa
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

stage_dict = {
    'N1': 0,
    'N2': 1,
    'N3': 2,
    'REM': 3,
    'WK': 4,
}


class SnoringDataset(Dataset):
    def __init__(self, label_file, dataset_path):
        self.labels = pd.read_csv(label_file)
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.dataset_path, f'{self.labels.iloc[idx, 0]}.wav')
        audio = librosa.load(audio_path, sr=16000, mono=False)
        label = self.labels.iloc[idx, 1]
        label_translated = stage_dict.get(label, '5')
        return audio, label_translated

    @staticmethod
    def get_targets(label_file, indices):
        """
        get targets list of dataset
        :param label_file: the original label file, which contains labels of both train and test set
        :param indices: an attribute of train or test dataset, because the original dataset has been
                        split randomly, indices[i] is the audio's index in the original dataset sequence
        :return: targets list of current dataset
        """
        labels = pd.read_csv(label_file)
        targets = np.zeros(len(indices), dtype=int)
        for i in range(len(indices)):
            targets[i] = stage_dict[labels.iloc[indices[i], 1]]
        return targets


# print the label distribution of dataset,
# this function won't use in the training process
def get_class_distribution():
    subject_path = 'F:\\Dataset'
    subjects = os.listdir(subject_path)
    count_all_dict = {k: 0 for k, v in stage_dict.items()}
    percent_all_dict = {k: " " for k, v in stage_dict.items()}
    for subject in subjects:
        dataset = SnoringDataset(
            label_file=f'{subject_path}\\{subject}\\SleepStaging.csv',
            dataset_path=f'{subject_path}\\{subject}\\Snoring_16k')
        idx2class = {v: k for k, v in stage_dict.items()}
        count_dict = {k: 0 for k, v in stage_dict.items()}
        percent_dict = {k: " " for k, v in stage_dict.items()}
        for element in dataset:
            y_lbl = element[1]
            y_lbl = idx2class[y_lbl]
            count_dict[y_lbl] += 1
        sum_ = 0
        for k, v in count_dict.items():
            sum_ += v
        for k, v in count_dict.items():
            percent_dict[k] = f"{(v / sum_) * 100:>0.1f}%"
        for k, v in count_dict.items():
            count_all_dict[k] += v
        print(f"Subject: {subject}".center(80, '-'))
        print(count_dict)
        print(percent_dict)
    sum_ = 0
    for k, v in count_all_dict.items():
        sum_ += v
    for k, v in count_all_dict.items():
        percent_all_dict[k] = f"{(v / sum_) * 100:>0.1f}%"
    print("Data Distribution of All Subjects".center(80, '-'))
    print(count_all_dict)
    print(percent_all_dict)


# get the current data distribution of the current train or test set
# this function is used in the training process
def get_current_class_distribution(dataset):
    idx2class = {v: k for k, v in stage_dict.items()}
    count_dict = {k: 0 for k, v in stage_dict.items()}
    for element in dataset:
        y_lbl = element
        y_lbl = idx2class[y_lbl.item()]  # convert number label to string label, like 0 to "N1"
        count_dict[y_lbl] += 1
    return count_dict
