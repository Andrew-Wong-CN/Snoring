import os
import numpy as np
import torch
import numpy
from loss import loss_cross
from torch.utils.data import DataLoader, WeightedRandomSampler
from prepro import get_mel_phase_batch, concat_mel_and_phase
from prepro import separate_channels
from dataset import SnoringDataset
from models.stagingnet import StagingNet
from dataset import get_current_class_distribution

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

learning_rate = 1e-3
epochs = 10
batch_size = 16

model = StagingNet()
# model.load_state_dict(torch.load("model_1.pth"))
model.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

loss_fn = loss_cross


def train_loop(dataloader, train_model, train_loss_fn, optimizer):

    size = len(dataloader.dataset)

    # batch_idx is the index of current batch, the 1st batch, 2nd batch
    # x is the original data in dataset
    # y is label(s, if batch > 1) of current data
    for batch_idx, (x, y) in enumerate(dataloader):

        x = x[0].numpy()

        # separate two channels
        input_left, input_right = separate_channels(x)

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
        y = y.to(device)
        loss = train_loss_fn(pred1, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch_idx * batch_size
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, test_model, test_loss_fn):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    correct_class = np.zeros(5) # correct_class[0] represents the number of correct prediction of 'N1'
    size_class = np.zeros(5) # store the number of labels of each class

    with torch.no_grad():

        # x is the original data in dataset
        # y is label(s, if batch > 1) of current data
        # NOTE: if batch_size > 1, len(x) and len(y) equal to batch_size
        for x, y in dataloader:

            # separate audio into two channels, get mel magnitude spectrum and concatenate them
            x = x[0].numpy()
            input_left, input_right = separate_channels(x)
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

def main():

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

            # compute weights
            # targets_train is a tensor of target labels of train set, targets_test is the same
            targets_train = SnoringDataset.get_targets(label_file=f'{subject_path}\\{subject}\\SleepStaging.csv',
                                                                   indices=train_dataset.indices)
            targets_test = SnoringDataset.get_targets(label_file=f'{subject_path}\\{subject}\\SleepStaging.csv',
                                                                 indices=test_dataset.indices)
            class_count_train = [i for i in get_current_class_distribution(targets_train).values()]
            class_count_test = [i for i in get_current_class_distribution(targets_test).values()]
            class_weights_train = 1. / torch.tensor(class_count_train, dtype=torch.float32)
            class_weights_test = 1. / torch.tensor(class_count_test, dtype=torch.float32)
            weights_train = class_weights_train[targets_train]
            weights_test = class_weights_test[targets_test]

            # define the weighted random sampler
            # replacement=True means samples can be drawn in multiple times
            weighted_sampler_train = WeightedRandomSampler(weights=weights_train,
                                                           num_samples=len(train_dataset),
                                                           replacement=True)
            weighted_sampler_test = WeightedRandomSampler(weights=weights_test,
                                                          num_samples=len(test_dataset),
                                                          replacement=True)

            # load data
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      sampler=weighted_sampler_train)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     sampler=weighted_sampler_test)

            # train and test model
            train_loop(dataloader=train_loader, train_model=model, train_loss_fn=loss_fn, optimizer=optimizer)
            test_loop(test_loader,model,loss_fn)

    print("Done")


if __name__ == "__main__":
    main()
    torch.save(model.state_dict(), "model_3.pth")
    print("Saved PyTorch Model State to model_3.pth")