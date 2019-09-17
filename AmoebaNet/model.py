import torch
from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
import sys
import numpy as np

##########################################
# AmoebaNet-A parameters
# N = Normal cell iteration number
# F = number of ouput filters of the conv ops
##########################################

def separableConv2d(in_planes, out_planes, kernel_size=3, stride=1):
    if kernel_size == 3:
        pad = 1
    elif kernel_size == 5:
        pad = 2
    else:
        pad = 3
    x = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=pad, groups=in_planes, bias=False),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False))
    return x

class AmoebaNet(nn.Module):
    """
    # Function: AmoebaNet-A for cifar10
    # Arguments:
        N: normal cell iter
        F: filter outputs
        p: sceduled droppath rate
    """
    def __init__(self, N=6, F=32, p=0.7):
        super(AmoebaNet, self).__init__()
        self.N = N
        self.F = F
        self.p = p
        
    def _normal_cell(self, x, in_planes, out_planes):
        x0 = x
        x1 = x
        
        s2_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x0)
        s2_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(x0)
        s2 = s2_1 + s2_2
        
        s3_1 = x0
        s3_2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x1)
        s3 = s3_1 + s3_2
        
        s4_1 = separableConv2d(in_planes, out_planes, kernel_size=5, stride=1)(s2)
        s4_1 = nn.BatchNorm2d(out_planes)(s4_1)
        s4_1 = nn.ReLU(inplace=True)(s4_1)
        s4_2 = separableConv2d(in_planes, out_planes, kernel_size=3, stride=1)(x1)
        s4_2 = nn.BatchNorm2d(out_planes)(s4_2)
        s4_2 = nn.ReLU(inplace=True)(s4_2)
        s4 = s4_1 + s4_2
        
        s5_1 = separableConv2d(in_planes, out_planes, kernel_size=3, stride=1)(s2)
        s5_1 = nn.BatchNorm2d(out_planes)(s5_1)
        s5_1 = nn.ReLU(inplace=True)(s5_1)
        s5_2 = s2
        s5 = s5_1 + s5_2
        
        s6_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(s4)
        s6_2 = separableConv2d(in_planes, out_planes, kernel_size=3, stride=1)(x0)
        s6_2 = nn.BatchNorm2d(out_planes)(s6_2)
        s6_2 = nn.ReLU(inplace=True)(s6_2)
        s6 = s6_1 + s6_2
        
        s7 = torch.cat((s3, s5, s6), -1)
        return s7
    
    def _reduction_cell(self, x, in_planes, out_planes):
        x0 = x
        x1 = x
        
        s2_1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)(x0)
        s2_2 = separableConv2d(in_planes, out_planes, kernel_size=3, stride=2)(x1)
        s2_2 = nn.BatchNorm2d(out_planes)(s2_2)
        s2_2 = nn.ReLU(inplace=True)(s2_2)
        s2 = s2_1 + s2_2
        
        s3_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x1)
        s3_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x0)
        s3 = s3_1 + s3_2
        
        s4_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x0)
        s4_2 = separableConv2d(in_planes, out_planes, kernel_size=7, stride=2)(s2)
        s4_2 = nn.BatchNorm2d(out_planes)(s4_2)
        s4_2 = nn.ReLU(inplace=True)(s4_2)
        s4 = s4_1 + s4_2
        
        s5_1 = separableConv2d(in_planes, out_planes, kernel_size=7, stride=2)(x0)
        s5_1 = nn.BatchNorm2d(out_planes)(s5_1)
        s5_1 = nn.ReLU(inplace=True)(s5_1)
        s5_2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)(x1)
        s5 = s5_1 + s5_2
        
        s6_1 = separableConv2d(in_planes, out_planes, kernel_size=3, stride=2)(s3)
        s6_1 = nn.BatchNorm2d(out_planes)(s6_1)
        s6_1 = nn.ReLU(inplace=True)(s6_1)
        s6_2 = nn.Conv2d(in_planes, out_planes, kernel_size=(7, 1), stride=2, padding=1)(x0)
        s6_2 = nn.Conv2d(out_planes, out_planes, kernel_size=(1, 7), stride=2, padding=1)(s6_2)
        s6_2 = nn.BatchNorm2d(out_planes)(s6_2)
        s6_2 = nn.ReLU(inplace=True)(s6_2)
        s6 = s6_1 + s6_2
        
        s7 = torch.cat((s4, s5, s6), -1)
        return s7
    
    def forward(self, x):
        
        x = nn.Conv2d(3, self.F, )
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        