import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os

os.chdir("d:/github/Paper2Code/ResNet")

from model_pytorch import *
from torchsummary import summary

device = 'cuda:0'
check_point = False


best_acc = 0
start_epoch = 0
lr = 0.1

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4464), (0.2023, 0.1994, 0.2010)),])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4464), (0.2023, 0.1994, 0.2010)),])

trainset = CIFAR10('./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testset = CIFAR10('./data', train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
net = preact_resnet1001()
net = net.to(device)
summary(net, (3, 32, 32))

cudnn.benchmark = True

if check_point:
    print("Resuming CheckPoint")
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0 
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print("Loss: %.3f | Acc %.3f%% (%d/%d)" %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# Test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %(test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        print("Saving")
        state = {'net':net.state_dict(), 'acc':acc, 'epoch':epoch,}
        if not os.path.isdir('checkpoint'):
            os.mkidr('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    