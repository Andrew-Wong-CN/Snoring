import os
import numpy as np
import torch
import numpy
from loss import loss_cross
from torch.utils.data import DataLoader
from prepro import get_mel_phase_batch, concat_mel_and_phase
from prepro import separate_channels
from dataset import SnoringDataset
from models.sound_stage_net_v2 import SoundStageNetV2, SoundStageNetV2Test
from models.classifier import Classifier
from tabulate import tabulate

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

learning_rate = 1e-3
epochs = 50
batch_size = 16
batch_size_test = 8
print((learning_rate, epochs, batch_size))

# initialize model
model = SoundStageNetV2()
pretraining_model_state_dict = torch.load("parameters/pretraining_SSNV2.pth")
model_state_dict = model.state_dict()
for k, v in pretraining_model_state_dict.items():
    if "FeatureExtractor" in k and k in model_state_dict.items():
        model_state_dict[k] = pretraining_model_state_dict[k]
model.load_state_dict(model_state_dict)
model.to(device)
model_test = SoundStageNetV2Test()
model_test.to(device)
classifier = Classifier(in_features=400, out_features=5, mid_features=40)
classifier.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
loss_fn = loss_cross


def train_loop(dataloader, train_model, train_loss_fn, optimizer):

    size = len(dataloader.dataset)

    # batch_idx is the index of current batch, the 1st batch, 2nd batch
    # x is the original data in dataset
    # y is label(s, if batch > 1) of current data
    for batch_idx, (x, y) in enumerate(dataloader):

        x = x[0].numpy() # x[0] is audio data, x[1] is sampling rate

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
        if current % 64 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, test_model, test_loss_fn):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    correct_class = np.zeros(5) # correct_class[0] represents the number of correct prediction of 'N1'
    size_class = np.zeros(5) # store the number of labels of each class
    confusion_matrix = np.zeros((5, 5), dtype=int)

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

            # zero padding
            zeros = np.zeros((int(input_left_mel.shape[0] / 2), input_left_mel.shape[1], input_left_mel.shape[2]))
            input_left_mel = np.concatenate((zeros, input_left_mel, zeros), axis=0)
            input_right_mel = np.concatenate((zeros, input_right_mel, zeros), axis=0)
            data = concat_mel_and_phase((input_left_mel, input_right_mel))

            # convert array to Tensor and move data to device
            data_torch = torch.Tensor(numpy.asarray(data))
            data_torch = data_torch.to(device)
            y = y.to(device)

            # pred1 without softmax, pred2 with softmax
            pred= test_model(data_torch)
            pred = pred[4:12]
            pred1, pred2 = classifier(pred)
            test_loss += test_loss_fn(pred1, y).item()

            # compute the count of each label predicted correctly and the size of each label
            pred_label = torch.argmax(pred2, dim=1)
            for i in range(len(pred_label)):
                if pred_label[i] == y[i]:
                    correct += 1
                    correct_class[pred_label[i]] += 1
                size_class[y[i]] += 1

            # compute the confusion matrix
            for i in range(len(y)):
                confusion_matrix[y[i].item()][pred_label[i].item()] += 1

    # compute the average test loss and accuracy
    test_loss /= num_batches
    accuracy_all = correct / size
    accuracy = np.zeros(5)
    for i in range(len(size_class)):
        if size_class[i] == 0:
            size_class[i] = 1
    if not 0 in size_class:
        accuracy = correct_class / size_class
    accuracy = np.array([round(i, 2) for i in accuracy])
    print(f"Accuracy for all classes: {(100 * accuracy_all):>0.1f}%\nAvg loss: {test_loss:>8f}\n")

    # compute the precision and recall
    precision = np.zeros(5)
    recall = np.zeros(5)
    for i in range(5):
        if np.sum(confusion_matrix, axis=0)[i] != 0:
            precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=0)[i]
        if np.sum(confusion_matrix, axis=1)[i] != 0:
            recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=1)[i]
    precision = np.array([round(i, 2) for i in precision])
    recall = np.array([round(i, 2) for i in recall])

    # compute F1 score
    f1 = np.zeros(5, dtype=float)
    for i in range(5):
        if precision[i] + recall[i] != 0.:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    f1 = np.array([round(i, 2) for i in f1])

    # convert confusion matrix items to percentage
    sum_ = np.sum(confusion_matrix, axis=1)
    confusion_matrix = confusion_matrix.astype(float)
    for i in range(5):
        for j in range(5):
            if sum_[i] != 0:
                confusion_matrix[i][j] = round(confusion_matrix[i][j] / sum_[i], 2)
    index = ["TN1", "TN2", "TN3", "TREM", "TWK"]
    header = ["PN1", "PN2", "PN3", "PREM", "PWK"]
    print("Confusion Matrix".center(46))
    print(tabulate(confusion_matrix, headers=header, showindex=index, tablefmt="fancy_grid"))
    print(" ")

    # print test result
    index = ["Accuracy", "Precision", "Recall", "F1"]
    header = ["N1", "N2", "N3", "REM", "WK"]
    info = np.array([accuracy, precision, recall, f1])
    print("Performance Error".center(46))
    print(tabulate(info, headers=header, showindex=index, tablefmt="fancy_grid"))
    print(" ")

    return accuracy, precision, recall, f1, accuracy_all

def main():

    # subject means the data of each patient
    subject_path = '/data1/wqz/Dataset'
    print(os.listdir(subject_path))
    subjects = os.listdir(subject_path)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)

    # iterate train and test for 10 times
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        m = 0

        # initialize performance of current epoch
        acc = np.zeros(5)
        prc = np.zeros(5)
        rc = np.zeros(5)
        f1_ = np.zeros(5)
        acc_avg = 0.

        for subject in subjects:
            m += 1
            print(f"Current subject: {subject} (Subject {m} / {len(subjects)}, Epoch {t + 1} / {epochs})")

            # initialize dataset
            dataset = SnoringDataset(
                label_file=f'{subject_path}/{subject}/SleepStaging.csv',
                dataset_path=f'{subject_path}/{subject}/Snoring_16k')

            # split train set and test set
            train_test_size_file = open(f"{subject_path}/{subject}/TrainTestSize.txt", "r+")
            size = train_test_size_file.readlines()
            train_size = int(size[0])
            test_size = int(size[1])
            train_test_size_file.close()
            train_indices = []
            test_indices = []
            indices = np.arange((train_size + test_size))
            for i in range(int(len(dataset) / batch_size)):
                if i % 3 == 0 and (i / 3) * 16 < test_size: # split into test
                    test_indices = np.concatenate((test_indices, indices[i * batch_size:(i + 1) * batch_size]), axis=0)
                else:
                    train_indices = np.concatenate((train_indices, indices[i * batch_size:(i + 1) * batch_size]), axis=0)
            train_indices.reshape(train_size)
            test_indices.reshape(test_size)
            train_indices = train_indices.astype("int64")
            test_indices = test_indices.astype("int64")
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

            # load data
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size_test,
                                     shuffle=False)

            # train and test model
            train_loop(train_loader, model, loss_fn, optimizer)
            accuracy, precision, recall, f1, accuracy_all = test_loop(test_loader, model_test, loss_fn)

            # compute overall performance
            acc += accuracy
            prc += precision
            rc += recall
            f1_ += f1
            acc_avg += accuracy_all

        # print overall performance error of current epoch
        len_subj = len(subjects)
        acc /= len_subj
        acc = np.append(acc, sum(acc)/len(acc))
        acc = np.array([round(i, 2) for i in acc])
        prc /= len_subj
        prc = np.append(prc, sum(prc)/len(prc))
        prc = np.array([round(i, 2) for i in prc])
        rc /= len_subj
        rc = np.append(rc, sum(rc)/len(rc))
        rc = np.array([round(i, 2) for i in rc])
        f1_ /= len_subj
        f1_ = np.append(f1_, sum(f1_)/len(f1_))
        f1_ = np.array([round(i, 2) for i in f1_])
        index = ["Accuracy", "Precision", "Recall", "F1"]
        header = ["N1", "N2", "N3", "REM", "WK", "AVG"]
        info = np.array([acc, prc, rc, f1_])
        print("Overall Performance Error".center(50))
        print(tabulate(info, headers=header, showindex=index, tablefmt="fancy_grid"))
        print(f"Average Accuracy of Current Epoch: {acc_avg/len_subj}")
        print(" ")

    print("Done")


if __name__ == "__main__":
    main()
    save_path = "parameters/training_SSNV1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved PyTorch Model State to {save_path}")