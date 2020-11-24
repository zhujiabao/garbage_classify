import torch
import torch.nn as nn


def CrossEntropyLoss_label_smooth(outputs, targets, num_classes=40, epsilon=0.1):
    N = targets.size(0)
    # torch.Size([8, 10])
    # 初始化一个矩阵, 里面的值都是epsilon / (num_classes - 1)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))

    targets = targets.data
    # 为矩阵中的每一行的某个index的位置赋值为1 - epsilon
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)
    # 调用torch的log_softmax
    log_prob = nn.functional.log_softmax(outputs, dim=1)
    # 用之前得到的smoothed_labels来调整log_prob中每个值
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss

