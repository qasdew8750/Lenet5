
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
import cv2
import matplotlib.pyplot as plt

seed = 42
Learning_rate = 0.001
epochs = 10
im_size = 32
N_classes = 10
file = "./file"
batch_size = 64
show_loss = True
showt_accuracy = True

## Setting of ResNet
out_ch = [64, 128, 256, 512]
num_blocks = [2, 2, 2, 2]


## declaration of Nets

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.conv3 = nn.Conv2d(16,120,5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = F.max_pool2d(F.relu(self.conv1(x)),2)
        y = F.max_pool2d(F.relu(self.conv2(y)),2)
        y = F.relu(self.conv3(y))
        y = y.view(-1, 120)
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

class LeNet5_with_BN(nn.Module):
    def __init__(self):
        super(LeNet5_with_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.bn3 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 120)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

class BasicBlock(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(BasicBlock, self).__init__()
    self.a = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel)
    )
    if(in_channel != out_channel):
      print(7)
      self.b = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel)
      )
    else:  
      print(6)
      self.b = nn.Sequential()
  def forward(self, x):
    out = self.a(x) + self.b(x)
    out = F.relu(out)
    return out

class BottleNeck(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(BasicBlock, self).__init__()
    self.a = nn.Sequential(
      nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channel, out_channel * 4, kernel_size=1, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.ReLU(inplace=True)
    )
    self.b = nn.Sequential()
  def forward(self, x):
    out = self.a(x)
    out += x
    return out

class ResNet(nn.Module):
  def __init__(self,block, out_ch, num_blocks, classes = 10):
      super(ResNet, self).__init__()
      self.initial_layer = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
      )
      self.tracking_ch = 64
      layers = []
      k = 0
      for i in num_blocks:
        print(self.tracking_ch, out_ch[k])
        layers.append(self.make_blocks(block,out_ch[k], num_blocks[i]))
        k += 1
      self._layers = nn.Sequential(*layers)
      self.final_layer = nn.Linear(512, classes)
  def make_blocks(self, block, out_ch, num_block, stride=1):
        layers = []
        layers.append(block(self.tracking_ch, out_ch))
        self.tracking_ch = out_ch
        for j in range(num_block - 1):
            layers.append(block(self.tracking_ch,out_ch))
            self.tracking_ch = out_ch
        return nn.Sequential(*layers)
  def forward(self, x):
    out = self.initial_layer(x)
    print(1)
    print(out.size())
    out = self._layers(out)

    print(4)
    print(out.size(0))
    out = F.avg_pool2d(out, out.size()[3])
    out = out.view(out.size(0), -1)
    print(5,out.size())
    out = self.final_layer(out)
    return out

## Setting of Net
batch_normalize = True
use_Resnet = True

if(0):
    net = LeNet5_with_BN()
elif(0):
    net = LeNet5()
else:
    net =  ResNet(BasicBlock, out_ch, num_blocks)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])
optimizer = optim.Adam(net.parameters(), lr = Learning_rate)
criterion = nn.CrossEntropyLoss()

## Setting of datasets
train_set = datasets.CIFAR10(root=file,
                                 train=True,
                                 download=True,
                                 transform=transform)

test_set = datasets.CIFAR10(root=file,
                                train=False,
                                download=True,
                                transform=transform)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

def train(epoch):
    net.train()
    loss_list= []
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        output = net(images)
        loss = criterion(output, labels)
        loss_list.append(loss.detach().cpu().item())

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))


        loss.backward()
        optimizer.step()

@torch.no_grad()
def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum().item()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss /= len(test_set)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss, float(total_correct) / len(test_set)))


def train_and_test(epoch):
    train(epoch)
    test()

for e in range(1, epochs + 1):
    train_and_test(e)
