# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月21日
"""

import os

import numpy as np
import torch
import numpy
from loss import loss_cross, one_hot
from torch.utils.data import DataLoader
from prepro import get_mel_phase_batch, concat_mel_and_phase
from prepro import separate_channels
from dataset import SnoringDataset
from models.Staging import Staging


device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

learning_rate = 1e-3
epochs = 10
batch_size = 16

model = Staging()
# model.load_state_dict(torch.load("model_1.pth"))
# model.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)

loss_fn = loss_cross


def train_loop(dataloader, train_model, train_loss_fn, optimizer):

    size = len(dataloader.dataset)
    for batch, (input_, label) in enumerate(dataloader):

        input_ = input_[0].numpy()

        # separate two channels
        input_left, input_right = separate_channels(input_)

        # get mel-spectrogram and phase, only use mel now!!!
        input_left_mel, input_left_phase = get_mel_phase_batch(input_left)
        input_right_mel, input_right_phase = get_mel_phase_batch(input_right)

        # concatenate mag1, mag2
        data = concat_mel_and_phase((input_left_mel, input_right_mel))

        # transform data from array to Tensor
        data_torch = torch.Tensor(numpy.asarray(data))
        data_torch = data_torch.to(device)

        # compute prediction and loss
        pred1, pred2 = train_model(data_torch) # pred1 without softmax, pred2 with softmax
        pred1 = pred1.to(device)
        label = label.to(device)
        loss = train_loss_fn(pred1, label)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * batch_size
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, test_model, test_loss_fn):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    correct_class = np.zeros(5) # correct_class[0] represents the number of correct prediction of 'N1'
    size_class = np.zeros(5) # store the number of labels of each class


    with torch.no_grad():
        for X, y in dataloader:

            # separate audio into two channels, get mel magnitude spectrum and concatenate them
            X = X[0].numpy()
            input_left, input_right = separate_channels(X)
            input_left_mel, input_left_phase = get_mel_phase_batch(input_left)
            input_right_mel, input_right_phase = get_mel_phase_batch(input_right)
            data = concat_mel_and_phase((input_left_mel, input_right_mel))

            # convert array to Tensor and move data to device
            data_torch = torch.Tensor(numpy.asarray(data))
            data_torch = data_torch.to(device)
            y = y.to(device)

            # pred1 without softmax, pred2 with softmax
            pred1, pred2 = test_model(data_torch)
            test_loss += test_loss_fn(pred1, y).item()

            # compute the count of each label predicted correctly and the size of each label
            pred_label = torch.argmax(pred2, dim=1)
            for i in range(len(pred_label)):
                if pred_label[i] == y[i]:
                    correct += 1
                    correct_class[pred_label[i]] += 1
                size_class[y[i]] += 1

    # compute the average test loss and accuracy
    test_loss /= num_batches
    correct /= size
    accuracy = np.zeros(5)
    for i in range(len(size_class)):
        if size_class[i] == 0:
            size_class[i] = 1
    if not 0 in size_class:
        accuracy = correct_class / size_class

    print(f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    print(f"Accuracy for N1 {(100 * accuracy[0]):>0.1f}%")
    print(f"Accuracy for N2 {(100 * accuracy[1]):>0.1f}%")
    print(f"Accuracy for N3 {(100 * accuracy[2]):>0.1f}%")
    print(f"Accuracy for REM {(100 * accuracy[3]):>0.1f}%")
    print(f"Accuracy for WK {(100 * accuracy[4]):>0.1f}% \n")
    print("--------------------")

def train():

    # subject means the data of each patient
    subject_path = 'F:\\Dataset'
    print(os.listdir(subject_path))
    subjects = os.listdir(subject_path)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    # iterate train and test for 10 times
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        for subject in subjects:
            print(f"Current subject: {subject}")

            # initialize dataset
            dataset = SnoringDataset(
                label_file=f'{subject_path}\\{subject}\\SleepStaging.csv',
                dataset_path=f'{subject_path}\\{subject}\\Snoring_16k')

            # split train set and test set
            train_test_size_file = open(f"{subject_path}\\{subject}\\TrainTestSize.txt", "r+")
            size = train_test_size_file.readlines()
            train_size = int(size[0])
            test_size = int(size[1])
            train_test_size_file.close()
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

            # load data
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # train and test model
            train_loop(dataloader=train_loader, train_model=model, train_loss_fn=loss_fn, optimizer=optimizer)
            test_loop(test_loader,model,loss_fn)

    print("Done")

def get_class_distribution():
    subject_path = 'F:\\Dataset'
    subjects = os.listdir(subject_path)
    stage_dict = {
        # 单向map
        'N1': 0,
        'N2': 1,
        'N3': 2,
        'REM': 3,
        'WK': 4,
    }
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


if __name__ == "__main__":
    # train()
    # torch.save(model.state_dict(), "model_2.pth")
    # print("Saved PyTorch Model State to model_2.pth")
    get_class_distribution()