"""
##############################################################################
# ResNet v2: Identity Mappings in Deep Residual Networks
# https://arxiv.org/pdf/1603.05027.pdf
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
##############################################################################
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

class PreActBlock(nn.Module):
    """
    # Function: PreActBlock
                BN - ReLU - 3x3Conv - BN - ReLU - 3x3Conv
    # Arguments:
        in_planes: input channel dims
        planes: output channel dims
        stride: 1
    
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
    
    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out
    
class PreActBottleneckBlock(nn.Module):
    """
    # Function: PreActBlock_Bottleneck
                BN - ReLU - 3x3Conv - BN - ReLU - 3x3Conv
    # Arguments:
        in_planes: input channel dims
        planes: output channel dims
        stride: 1
    
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckBlock, self).__init__()
        bottleneck_planes = planes // 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, bottleneck_planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(bottleneck_planes)
        self.conv2 = nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(bottleneck_planes)
        self.conv3 = nn.Conv2d(bottleneck_planes, self.expansion*planes, kernel_size=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
    
    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv3(out)
        out += shortcut
        return out    

    
class PreActResNet(nn.Module):
    """
    # Function: Pre-activation ResNet v2
        BN - ReLU - Conv2D - BN - ReLU - Conv - ADD
        conv1 :  32x32, 16
        stage 0: 32x32, 64
        stage 1: 16x16, 128
        stage 2:  8x8,  256
    # Arguments:
        block: stage
        block_num: block_num
        num_classes: cifar10 = 10
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        """
        # Function: Stage building
        # Arguments:
            block: previous block
            planes: input channel dims
            num_blocks: number of inner block
            stride
        """
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def preact_resnet1001():
    return PreActResNet(PreActBottleneckBlock, [111, 111, 111])