# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月21日
"""

import os
import time

import torch
import torch.nn as nn
import numpy
from torch import Tensor

from loss import loss_cross
from torch.utils.data import DataLoader
from prepro import get_mel_phase, get_mel_phase_batch, concat_mel_and_phase
from prepro import separate_channels
from dataset import SnoringDataset
from models.Staging import Staging

learning_rate = 1e-3
epochs = 10
batch_size = 128

model = Staging()
loss_fn = loss_cross
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    #TODO device
    size = len(dataloader.dataset)
    for batch, (input, label) in enumerate(dataloader):
        input = input[0].numpy()
        #  separate two channels
        input_left, input_right = separate_channels(input)

        # get mel-spectrogram and phase
        input_left_mel, input_left_phase = get_mel_phase_batch(input_left)
        input_right_mel, input_right_phase = get_mel_phase_batch(input_right)

        # # copy on GPU
        # input_left_mel = input_left_mel.to(device)
        # input_left_phase = input_left_phase.to(device)
        # input_right_mel = input_right_mel.to(device)
        # input_right_phase = input_right_phase.to(device)

        # 拼接mag1, mag2, phase1, phase2
        data = concat_mel_and_phase((input_left_mel, input_right_mel, input_left_phase, input_right_phase))
        # numpy.ndarray转torch
        data_torch = torch.Tensor(numpy.asarray(data))
        # compute prediction and loss
        pred = model(data_torch)
        loss = loss_fn(pred, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train():
    # device = 'cuda'
    # 遍历每个subject
    subject_path = 'D:\\Ameixa\\学习\\实验室\\Snoring Detection\\DataSet\\'
    print(os.listdir(subject_path))
    subjects = os.listdir(subject_path)
    # 初始化模型、优化器
    # model = Staging().to(device)
    # TODO: weight_decay, lr参数修改
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        for subject in subjects:
            # 初始化数据加载
            dataset = SnoringDataset(
                label_file=f'{subject_path}\\{subject}\\SleepStaging.csv',
                dataset_path=f'{subject_path}\\{subject}\\Snoring'
            )
            train_size = int(len(dataset) * 0.7)
            test_size = int(len(dataset) * 0.3)
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            #  TODO: shuffle=False
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            train_loop(dataloader=train_loader,model=model,loss_fn=loss_fn,optimizer=optimizer)
            # ,device = 'cuda:0'
            test_loop(test_loader,model,loss_fn)
    print("Done")


if __name__ == "__main__":
    train()