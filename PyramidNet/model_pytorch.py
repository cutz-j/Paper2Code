"""
##############################################################################
# PyramidNet: Deep Pyramidal Residual Networks
# https://arxiv.org/pdf/1610.02915v4.pdf
# PyramidNet(bottleneck, a=200| 26.0M| -----     | 3.31 +- 0.08  |
##############################################################################
"""
#############################################
# PyramidNet parameters
# alpha = widening factor
# N_num = total number of layers
#############################################

import torch
from torch import nn, optim
from torch.nn import functional as F

class BottleneckBlock(nn.Module):
    """
    # Function: Bottleneck Block
                BN - 1x1Conv - BN - ReLU - 3x3Conv - BN - ReLU - 1x1Conv - BN
    # Arguments:
        in_planes: input channel dims
        planes: output channel dims
        stride: 1
    
    """
    outchannel_ratio = 4
    expansion_ratio = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*self.expansion_ratio, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion_ratio)
        self.conv3 = nn.Conv2d(planes*self.expansion_ratio, planes*self.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes*self.outchannel_ratio)
        self.stride = stride
        
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv3(out)
        
        out = self.bn4(out)
        
        if self.stride != 1:
            shortcut = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
            shortcut = shortcut(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]
        
        batch_size = out.size()[0]
        res_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        
        if res_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, res_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut
        return out
    

class PyramidNet(nn.Module):
    """
    # Function: PyramidNet
        BN - 1x1Conv - BN - ReLU - 3x3Conv - BN - ReLU - 1x1Conv - BN - ADD
        conv1 :  32x32, 16
        stage 0: 32x32, 64
        stage 1: 16x16, 128
        stage 2:  8x8,  256
    # Arguments:
        block: stage
        num_classes: cifar10 = 10
    """
    def __init__(self, depth, alpha, num_classes=10):
        super(PyramidNet, self).__init__()
        self.in_planes = 16
        n = int((depth - 2)/9)
        block = BottleneckBlock
        self.add_rate = alpha / (3*n*1.0)
        
        self.input_feature_dim = self.in_planes
        self.conv1 = nn.Conv2d(3, self.input_feature_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_feature_dim)
        
        self.featuremap_dim = self.input_feature_dim
        self.stage1 = self._make_layer(block, n)
        self.stage2 = self._make_layer(block, n, stride=2)
        self.stage3 = self._make_layer(block, n, stride=2)
        
        self.final_feature_dim = self.input_feature_dim
        self.bn_final = nn.BatchNorm2d(self.final_feature_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.linear = nn.Linear(self.final_feature_dim, num_classes)
    
    def _make_layer(self, block, num_blocks, stride=1):
        """
        # Function: Stage building
        # Arguments:
            block: previous block
            num_blocks: number of inner block
            stride
        """
        layers = []
        # pyramid height increase like pyramid
        self.featuremap_dim = self.featuremap_dim + self.add_rate
        layers.append(block(self.input_feature_dim, int(round(self.featuremap_dim)), stride))
        for i in range(1, num_blocks):
            temp_featuremap_dim = self.featuremap_dim + self.add_rate
            layers.append(block(int(round(self.featuremap_dim))*block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim = temp_featuremap_dim
        self.input_feature_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        
        out = self.bn_final(out)
        out = F.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

