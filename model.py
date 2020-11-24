#from efficientnet_pytorch import EfficientNet
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small, h_swish
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
write = SummaryWriter("runs/log")


class garbageModel():
    def __init__(self, lr, pretrained=True, numclass=40):
        #是否加载预训练模型
        if pretrained:
            self.model = mobilenetv3_small()
            self.model.load_state_dict(torch.load('mobilenetv3-small-55df8e1f.pth'))

            self.model.classifier = nn.Sequential(
                                        nn.Linear(576, 1024),
                                        h_swish(),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, numclass),
        )
        else:
            self.model = mobilenetv3_small(num_classes=numclass)

        self.model.cuda()
        self.numclass = numclass

        #多少分类任务

        self.optimizer = optim.SGD(self.model.parameters() , lr=lr, momentum=0.9)
        self.exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                               T_0=5)
        #print(self.model)

    def train(self, trainLoder, epoch, criterion):
        total = 0.
        running_loss = 0.
        correct = 0.

        self.model.train()

        for i, data in enumerate(tqdm(trainLoder)):
            img, label = data
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()

            self.optimizer.zero_grad()

            result = self.model(img)

            loss = criterion(result, label)

            #back
            loss.backward()
            self.optimizer.step()
            self.exp_lr_scheduler.step()

            _, predicted = torch.max(result.data, 1)

            total += label.size(0)
            running_loss += loss.data.item()
            correct += (predicted==label).sum()
            write.add_scalar("Train loss", loss.data.item(), epoch*len(trainLoder) + i)
            #打印log
            if (i+1)%20 == 0:
                print("Train Epoch:{}, iter:{} ,Loss:{:.4f}".format(epoch, i ,loss.data.item()))

        print("Train Epoch:{} finished, loss_mean:{:.4f}, acc:{:.4f}".format(epoch, running_loss/total , correct/total))
        write.add_scalar("Train acc", correct/total, epoch)

        return correct/total, running_loss/total

    def valid(self, valLoader, epoch, criterion):
        torch.no_grad()
        self.model.eval()
        total = 0.
        running_loss = 0.
        correct = 0.

        for i, data in enumerate(tqdm(valLoader)):
            img, label = data
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()

            result = self.model(img)
            val_loss = criterion(result, label)

            _, predicted = torch.max(result.data, 1)

            total += label.size(0)
            running_loss += val_loss.data.item()
            correct += (predicted==label).sum()

            write.add_scalar("Valid loss", val_loss.data.item(), epoch*len(valLoader) + i)
            if (i + 1) % 20 == 0:
                print("Valid Epoch:{}, iter:{} ,Loss:{:.4f}".format(epoch, i, val_loss.data.item()))

        print("Valid Epoch:{} finished, loss_mean:{:.4f}, acc:{:.4f}".format(epoch, running_loss / total,
                                                                             correct / total))
        write.add_scalar("Valid acc", correct/total, epoch)

        return correct/total, running_loss/total


if __name__ == '__main__':
    model = garbageModel(pretrained=True, optimizer=False, exp_lr_scheduler=False)
    #from torchsummary import summary
    #summary(model.model, (3, 456,456), device='cpu')
    import time

    time1 = time.time()
    tet = torch.Tensor(1, 3, 456, 456)
    print(model.model(tet))
    print(time.time() - time1)
