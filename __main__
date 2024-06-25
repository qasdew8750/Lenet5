
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

seed = 42
Learning_rate = 0.01
size = 32
epochs = 10
im_size = 32
N_classes = 10
file = "./file"
batch_normalize = False
batch_size = 32



class LeNet5(nn.Module):
    def __init__(self,n_classes):
        super().__init__()  
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = F.max_pool2d(F.relu(self.conv1(x)))
        y = F.max_pool2d(F.relu(self.conv2(y)))
        y = torch.flatten(y)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

class LeNet5_with_BN(nn.Module):
    def __init__(self,n_classes):
        super().__init__()  
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = F.max_pool2d(F.relu(self.conv1(x)))
        y = F.max_pool2d(F.relu(self.conv2(y)))
        y = torch.flatten(y)
        y = F.relu(torch.nn.BatchNorm1d(self.fc1(y)))
        y = F.relu(torch.nn.BatchNorm1d(self.fc2(y)))
        y = self.fc3(y)
        return y

if(batch_normalize == True):
    net = LeNet5_with_BN(nn)
else:
    net = LeNet5()
optimizer = optim.SGD(net.parameters(), lr = Learning_rate)
criterion = nn.MSELoss()

train_set = datasets.CIFAR10(root=file,
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())

test_set = datasets.CIFAR10(root=file,
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())

 
# train_loader, test_loader 생성
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


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        avg_loss /= len(data_test)
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()

for e in range(1, epochs + 1):
    train_and_test(e)

