import sys
import os
from model import base
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
from utils.tools import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader
from datafolder import garbageData, load_dataset
from sklearn.model_selection import KFold
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small, h_swish


EPOCH = 10
LR = 0.01
BATCH_SIZE = 128
ROOT_IMG = './data/garbage_classify/train_data'
K = 7
PRETRAINED = True
NUM_CLASS = 40
start_epoch = 0

accurancy_global = 0.0
val_accurancy_global = 0.0
#加载数据
dataset = garbageData(img_root=ROOT_IMG)
#加载模型
if PRETRAINED:
    model = mobilenetv3_small()
    model.load_state_dict(torch.load('mobilenetv3-small-55df8e1f.pth'))

    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),
        h_swish(),
        nn.Dropout(0.2),
        nn.Linear(1024, NUM_CLASS),
    )
else:
    model = mobilenetv3_small(num_classes=NUM_CLASS)

if torch.cuda.is_available():
    model.cuda()

#
base = base()
# 多少分类任务

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                       T_0=5)
# print(self.model)
loss = LabelSmoothingCrossEntropy()

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
        curr_epoch = start_epoch + k*EPOCH + epoch + 1
        #训练
        train_correct, train_running_loss = base.train(model=model, trainLoder=train_loader, epoch=curr_epoch, criterion=loss, optimizer=optimizer,
                                                        exp_lr_scheduler=exp_lr_scheduler)
        #验证
        val_correct, val_running_loss = base.valid(model=model, valLoader=val_loader, epoch=curr_epoch, criterion=loss)
        #保存模型
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        k += 1
        if train_correct > accurancy_global:
            torch.save(state, "./weights/best.pkl")
            print("准确率由：", accurancy_global, "上升至：", train_correct, "已更新并保存权值为weights/best.pkl")
            accurancy_global = train_correct
torch.save(model, "./weights/final.pth")