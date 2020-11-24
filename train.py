import sys
import os
from model import garbageModel
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datafolder import garbageData, load_dataset
from sklearn.model_selection import KFold


EPOCH = 10
LR = 0.01
BATCH_SIZE = 128
ROOT_IMG = ''
K = 7
PRETRAINED = True
NUM_CLASS = 40
start_epoch = 0

accurancy_global = 0.0
val_accurancy_global = 0.0
#加载数据
dataset = garbageData(img_root=ROOT_IMG)
#加载模型
model = garbageModel(lr=LR, pretrained=PRETRAINED, numclass=NUM_CLASS)


#交叉验证
KF = KFold(K, shuffle=True, random_state=5)
k = 0

for train_index, val_index in KF.split(dataset):

    train_set = []
    valid_set = []

    for i in train_index:
        train_set.append(dataset[i])
    print("success train_Set size", len(train_set))
    for j in val_index:
        valid_set.append(dataset[j])
    print("success val_set size", len(valid_set))

    trainset = load_dataset(data_set=train_set)
    validset = load_dataset(data_set=valid_set)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4, drop_last=True)

    #
    for epoch in range(EPOCH):
        curr_epoch = start_epoch + K*EPOCH + epoch + 1
        #训练
        train_correct, train_running_loss = model.train(trainLoder=train_loader, epoch=curr_epoch)
        #验证
        val_correct, val_running_loss = model.valid(valLoader=val_loader, epoch=curr_epoch,)
        #保存模型
        state = {'net': model.model.state_dict(), 'optimizer': model.optimizer.state_dict(), 'epoch': epoch}

        if train_correct > accurancy_global:
            torch.save(state, "./weights/best.pkl")
            print("准确率由：", accurancy_global, "上升至：", train_correct, "已更新并保存权值为weights/best.pkl")
            accurancy_global = train_correct