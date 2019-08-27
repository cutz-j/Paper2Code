import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
import sys
import time
os.chdir("d:/github/Paper2Code/WideResNet")
from wrn import WideResNet

device = 'cuda:0'
start_epoch = 1
num_epochs = 200
batch_size = 128
optim_type = 'SGD'
num_classes = 10
root_dir = 'd:/dataset/cifar10'
d = 28 #
k = 10

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

best_acc = 0

### Augmentation ###
transform_train = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std),])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=mean, std=std),])

trainset = CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
testset = CIFAR10(root=root_dir, train=False, download=True, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=100, shuffle=False)

net = WideResNet(depth=d, widen_factor=k)
file_name = 'wide-resnet-'+str(d)+'x'+str(k)

## train ##

criterion = nn.CrossEntropyLoss()

for epoch in range(start_epoch, start_epoch+num_epochs):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=learning_rate(0.1, epoch), 
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, learning_rate(0.1, epoch)))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        print('| Epoch [%3d/%3d] Iter [%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' 
              %(epoch, num_epochs, batch_idx+1))

