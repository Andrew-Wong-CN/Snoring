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