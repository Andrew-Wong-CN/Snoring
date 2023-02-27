# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2023年02月26日
"""
import os.path
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

BATCH_SIZE = 256
TRAIN_SIZE = 124187
TEST_SIZE = 53223


class MelDataset(Dataset):
    def __init__(self, csv_path, npy_path):
        self.csv_path = csv_path
        self.npy_path = npy_path
        self.npy_list = os.listdir(self.npy_path)
        self.csv_list = os.listdir(self.csv_path)

    def __len__(self):
        return len(self.npy_list)

    def __getitem__(self, idx):
        mel_path = os.path.join(self.npy_path, self.npy_list[idx])
        mel = np.load(mel_path)
        label_path = os.path.join(self.csv_path, self.csv_list[idx])
        label = pd.read_csv(label_path)
        label = label["label"].values
        label = np.array(label, dtype=np.float32)
        label = label.reshape(label.shape[0])
        return mel, label

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(47*20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 47),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.mlp(x)
        return x

MLP = MLP().to(device)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 256 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MLP.parameters(), lr=1e-3)

def main():
    npy_path = "F:\\GeneratedData\\npy"
    csv_path = "F:\\GeneratedData\\csv"
    # load data
    dataset = MelDataset(csv_path=csv_path, npy_path=npy_path)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [TRAIN_SIZE, TEST_SIZE])
    mpssc_train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    mpssc_test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #
    # train_mels, train_labels = next(iter(mpssc_train_loader))
    # test_mels, test_labels = next(iter(mpssc_test_loader))
    # # train and test the mlp
    # mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4,
    #                     hidden_layer_sizes=(100, 47), random_state=1,
    #                     verbose=10, learning_rate_init=.1).fit(train_mels, train_labels)
    # print(mlp.score(test_mels, test_labels))
    # print(mlp.n_iter_)
    # print(mlp.loss_)
    # print(mlp.out_activation_)
    #
    # # predict on the original audio and store it
    # pre = mlp.predict()
    # np.savetxt('snore_detection.csv', pre, delimiter=',')
    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(mpssc_train_loader, MLP, loss, optimizer)
        test(mpssc_test_loader, MLP, loss)
    print("Done!")


if __name__ == '__main__':
    main()
