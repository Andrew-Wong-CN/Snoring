# category = torch.arange(0, 5, 1)
# category = category.to(device)
# pred_1 = torch.sum(torch.mul(category, pred), dim=-1)  # ??? 算加权平均，平均是哪个类，如果哪个类的概率高则偏向哪个类
# pred_2 = torch.round(pred_1)
# pred_2 = pred_2.int()
# # pred_2 = torch.reshape(pred_2, pred.size(1))
# for b in range(pred_2.size(0)): # size(0) is batch size, calculate the accuracy in each batch
#     # pred_2 size is (batch, frames) frames上所有frame对应预测标签的众数作为该segment对应的标签
#     t = torch.argmax(torch.bincount(pred_2[b]))
#     if t == y[b]:
#         correct += 1
#         if t.item() == 0:
#             correct_n1 += 1
#         elif t.item() == 1:
#             correct_n2 += 1
#         elif t.item() == 2:
#             correct_n3 += 1
#         elif t.item() == 3:
#             correct_rem += 1
#         elif t.item() == 4:
#             correct_wk += 1
# t1 = torch.bincount(y)
# size_n1 += t1[0].item()
# size_n2 += t1[1].item()
# size_n3 += t1[2].item()
# size_rem += t1[3].item()
# size_wk += t1[4].item()

# correct_n1, correct_n2, correct_n3, correct_rem, correct_wk = 0, 0, 0, 0, 0
# size_n1, size_n2, size_n3, size_rem, size_wk = 0, 0, 0, 0, 0

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# from torchsummary import summary

# target = one_hot(y, num_classes=5) # one-hot encoding labels
# target_argmax = torch.argmax(target, dim=1) # target_argmax represents the current label

# 将string类型label与数值映射
# WK:4   N1:0    N2:1    N3:2    REM:3
# label_encoder = preprocessing.LabelEncoder()
# label_encoder.fit(['WK', 'N1', 'N2', 'N3', 'REM'])
# label_transed = label_encoder.transform(label)
# label_encoder使用注意：输入必须是一个数组，相当于批量转换label

# import torch
#
# dataset = SnoringDataset(
#     label_file='D:\\Ameixa\\学习\\实验室\\Snoring Detection\\DataSet\\Subject0905\\SleepStaging.csv',
#     dataset_path='D:\\Ameixa\\学习\\实验室\\Snoring Detection\\DataSet\\Subject0905\\Snoring')
#
# print(len(dataset))
# train_size = int(len(dataset) * 0.7)
# test_size = int(len(dataset) * 0.3)
# # split the origin dataset randomly, which means the formal order is disturbed
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
# print(len(train_loader))
# print(len(test_loader))

#
# if __name__ == '__main__':
#     import os
#     files = os.listdir('F:\\Snore_Sound_Data\\2022-09-05-M-31\\segments1')
#     size0 = os.path.getsize('F:\\Snore_Sound_Data\\2022-09-05-M-31\\segments1' + '\\' + "0.wav")
#     print(size0)
#     size1 = os.path.getsize('F:\\Snore_Sound_Data\\2022-09-05-M-31\\segments1' + '\\' + "1.wav")
#     print(size1)
#     for file in files:
#         size = os.path.getsize('F:\\Snore_Sound_Data\\2022-09-05-M-31\\segments1' + '\\' + file)
#         if size != size1:
#             print(f"file: {file}, size: {size}")


# loading sound life
# y_32k, sr = librosa.load(fpath, sr=sample_rate_original)

# resampling
# y = librosa.resample(y, orig_sr=32000, target_sr=16000)

# Trimming causes the output is not same shape
# y, _ = librosa.effects.trim(y, top_db=top_db)

# shape = audio_data.shape
# left = audio_data[..., 0:shape[1]+1:2]
# right = audio_data[..., 1:shape[1]+1:2]

"""
separate stereo audio into mono audio
:param y: dual channel audio
:return: left and right channels
"""

"""
:param y: audio file (32k)
:return:
    mel: magnitude spectrogram, a 2d array of shape (frames, Mel frequency bins)
    phase: phase spectrogram
"""

    # category = torch.arange(0, 5, 1)
    # category = category.to(device)
    # one-hot encoding label
    # pred_1 = torch.sum(torch.mul(category,output),dim=-1) # ??? 算加权平均，平均是哪个类，如果哪个类的概率高则偏向哪个类
    # pred_1 = pred_1.to(device)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss = loss_fn(pred_1,label)

"""

:param input_: 输入数据的应该包含四个维度，第一个维度应该为声道数*2，
:return:
"""
# if input.dim() != 4:
#     raise (ValueError("expected 4D input, got {}D".format(input.dim())))
#
# if input.shape[1] % 2 != 0:
#     raise (ValueError("expected input 2nd dimension is even number, got {}".format(input.shape[0])))
# mid = input.shape[1] // 2 # '//' represents floor division. This // operator divides the first number by the second number and rounds the result down to the nearest integer (or whole number).
# mag = self.BatchNorm_mag(input[:, :mid, :, :])
# pha = self.BatchNorm_pha(input[:, mid:, :, :])
# input = torch.cat((mag, pha), dim=1)
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # model1 = Inception()
#     # model1 = model1.to(device)
#     # summary(model1, (2, 469, 20), 16)
#     # model2 = BlstmBlock(input_size=10, hidden_size=5)
#     # model2 = model2.to(device)
#     # summary(model2, (16, 469, 10))
#     model3 = Classification(in_features=4690, out_features=5, mid_features=469)
#     model3 = model3.to(device)
#     pytorch_total_params = sum(p.numel() for p in model3.parameters())
#     print(pytorch_total_params)

# # 用两个batchnorm分别对幅度谱，相位谱进行归一化处理, ba
# self.BatchNorm_mag = nn.BatchNorm2d(2)  # num features, represent C in (N, C, H, W)
# self.BatchNorm_pha = nn.BatchNorm2d(2)
# if __name__ == "__main__":
#     from torchinfo import summary
#
#     model = Inception()
#
#     # 4, 188, 257
#     summary(model, input_size=(4, 1, 2048), batch_size=32, device='cpu')

# if __name__ == '__main__':
#     from torchinfo import summary
#
#     model = BiLSTM(frequency=128)
#
#     summary(model, input_size=(356, 128), batch_size=2, device='cpu')

# print(f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
# print(f"Accuracy for N1 {(100 * accuracy[0]):>0.1f}%")
# print(f"Accuracy for N2 {(100 * accuracy[1]):>0.1f}%")
# print(f"Accuracy for N3 {(100 * accuracy[2]):>0.1f}%")
# print(f"Accuracy for REM {(100 * accuracy[3]):>0.1f}%")
# print(f"Accuracy for WK {(100 * accuracy[4]):>0.1f}%")
# print("----------------------------------------------")
# print(f"")
# info = {"N1": [accuracy[0], precision[0], recall[0], f1[0]],
#         "N2": [accuracy[1], precision[1], recall[1], f1[1]],
#         "N3": [accuracy[2], precision[2], recall[2], f1[2]],
#         "REM": [accuracy[3], precision[3], recall[3], f1[3]],
#         "WK": [accuracy[4], precision[4], recall[4], f1[4]]}