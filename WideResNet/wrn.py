import torch
from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
import sys
import numpy as np

##########################################
# Wide Residual Networks parameters
# k = widening factor in Resiudal network
# l = depth factor in Residual network
# d = number of Residual blocks
# n = total layers
##########################################

def conv3x3 (in_planes, out_planes, stride=1):
    """
    # Function: conv3x3 for pytorch
    # Arguments:
        - in_planes: in-channels
        - out_planes: out-channels
        - stride: strides
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    """
    # Function: weight initialization for Conv / BatchNorm
    # Arguments:
        m: layer
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    
class wide_basic(nn.Module):
    """
    # Function: Wide ResNet basic layer
        BN --> ReLU --> Conv --> Dropout // 3x3 --> 3x3
    # Arguments:
        in_planes: in-channels layer module
        planes: out-channels
        dropout_rate: 0.3 for cifar10
        stride
    """
    def __init__(self, in_planes, planes, dropout_rate=0.3, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes) # 2d input
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),)
        
    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out
    
class WideResNet(nn.Module):
    """
    # Function: WRN init
    # Arguments:
        depth: depth factor
        widen_factor: k is widen factor in residual network
        dropout_rate: dropout rate 0.3 for cifar10
        num_classes: num classses
    # Returns:
        out
    """
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10):
        super(Wide_ResNet, self).__init()
        self.in_planes = 16
        
        assert ((depth-4)%6 == 0)
        n = (depth-4)/6 # blocks num
        k = widen_factor
        
        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        
        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(block=wide_basic, planes=nStages[1], num_blocks=n, dropout_rate=dropout_rate, stride=1)
        self.layer2 = self._wide_layer(block=wide_basic, planes=nStages[2], num_blocks=n, dropout_rate=dropout_rate, stride=2)
        self.layer3 = self._wide_layer(block=wide_basic, planes=nStages[3], num_blocks=n, dropout_rate=dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        """
        # Function: wide layer for WRN
        # Arguments:
            block: residual block --> wide_basic
            planes: output-channels
            num_blocks: block numbers
            dropout_rate: 0.3
            stride
        """
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x) # stem
        out = self.layer1(out) # conv group 1
        out = self.layer2(out) # conv group 2
        out = self.layer3(out) # conv group 3
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8) # avg-pool
        out = out.view(out.size(0), -1) # flatten
        out = self.linear(out) # classfication layer
        return out