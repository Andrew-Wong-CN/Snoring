import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"


def loss_cross(output, label):
    target = one_hot(label, num_classes=5)
    target = target.to(device)
    log_softmax = nn.LogSoftmax(dim=1)
    pred = log_softmax(output)
    loss = - pred * target
    loss = torch.mean(torch.sum(loss, dim=1))
    return loss

# convert labels to one_hot
def one_hot(labels, num_classes):
    rows = labels.shape[0]
    output = torch.zeros((rows, num_classes))
    for row in range(rows):
        label = labels[row]
        output[row][label] = 1
    return output
