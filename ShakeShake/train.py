import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms, datasets
from shakeshake import *
from collections import OrderedDict
import random
import importlib

dataset_dir = 'd:/dataset/'
device = 'cuda:0'

EPOCHS = 1800
batch_size = 128
base_lr = 0.2
weight_decay = 1e-4
momentum = 0.9
lr_min = 0

## augmentation
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2470, 0.2435, 0.2616])

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])

train_dataset = datasets.CIFAR10(dataset_dir, train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR10(dataset_dir, train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device, drop_last=True,)
test_loader = DataLoader(test_dataset, batch_size=batch_size, nm_workers=0, shuffle=False, pin_memory=device, drop_last=False,)

## train
global_step = 0

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_cosine_annealing_scheduler(optimizer):
    total_steps = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: _cosine_annealing(step, total_steps, 1, lr_min / base_lr))
    return scheduler

def train(epoch, model, optimizer, scheduler, criterion, train_loader):
    global global_step
    
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1
        scheduler.step()
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        
        optimizer.step()
        _, preds = torch.max(outputs, dim=1)
        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)
        
        accuracy = correct_ / num
        loss_meter.update(loss_, num)
        acc_meter.update(accuracy, num)
        
        if step % 100 == 0:
            print('Epoch {} Step {}/{} Loss {:.4f} ({:.4f}) Accuracy {:.4f} ({:.4f})'.format(epoch, 
                  step, len(train_loader), loss_meter.val, loss_meter.avg, acc_meter.val, acc_meter.avg,))
    train_log = OrderedDict({'epoch': epoch, 
                             'train':OrderedDict({'loss':loss_meter.avg, 'accuracy':acc_meter.avg})})
    return train_log


def test(epoch, model, criterion, test_loader):
    model.eval()
    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, dim=1)
            
            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)
            
            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)
    acc = correct_meter.sum / len(test_loader.dataset)
    print('Epoch {%d} Loss {%.4f} Accuracy {%.4f}'.format(epoch, loss_meter.avg, acc))
    
    test_log = OrderedDict({'epoch':epoch, 
                            'test': OrderedDict({'loss':loss_meter.avg, 'accruacy':acc})})
    return test_log

# main
torch.manual_seed(7)
np.random.seed(7)
random.seed(7)

module = importlib.import_module('shake_shake')
Network = getattr(module, 'Network')
Network()














