#from efficientnet_pytorch import EfficientNet
import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
write = SummaryWriter("runs/log")


class base():

    def train(self, model, trainLoder, epoch, criterion,optimizer,exp_lr_scheduler):
        total = 0.
        running_loss = 0.
        correct = 0.

        model.train()

        for i, data in enumerate(tqdm(trainLoder)):
            img, label = data
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            #print(label)
            optimizer.zero_grad()

            result = model(img)

            loss = criterion(result, label)

            #back
            loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()

            _, predicted = torch.max(result.data, 1)
            #print(predicted)
            total += label.size(0)
            running_loss += loss.data.item()
            correct += (predicted==label).sum()
            write.add_scalar("Train loss", loss.data.item(), epoch*len(trainLoder) + i)
            write.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch*len(trainLoder) + i)
            #打印log
            if (i+1)%20 == 0:
                print("Train Epoch:{}, lr:{:.4f}, iter:{} ,Loss:{:.4f}".format(epoch, optimizer.param_groups[0]['lr'], i ,loss.data.item()))

        print("Train Epoch:{} finished, loss_mean:{:.4f}, acc:{:.4f}".format(epoch, running_loss/total , correct/total))
        write.add_scalar("Train acc", correct/total, epoch)

        return correct/total, running_loss/total

    def valid(self, model, valLoader, epoch, criterion):
        model.eval()
        total = 0.
        running_loss = 0.
        correct = 0.
        with torch.no_grad():
          for i, data in enumerate(tqdm(valLoader)):
              img, label = data
              if torch.cuda.is_available():
                  img = Variable(img).cuda()
                  label = Variable(label).cuda()
              result = model(img)
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
    model = base()
    #from torchsummary import summary
    #summary(model.model, (3, 456,456), device='cpu')
    import time

    time1 = time.time()
    tet = torch.Tensor(1, 3, 333, 333).cuda()
    print(model.model(tet))
    print(time.time() - time1)
