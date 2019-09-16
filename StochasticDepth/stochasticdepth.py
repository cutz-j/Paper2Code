from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import torch

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, strides=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strides, padding=1, bias=False)

def conv1x1(in_planes, out_planes, strides=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=strides, bias=False)

class StoDepth_BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, prob, multFlag, inplanes, planes, strides=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, strides)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.strides = strides
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag
        
    def forward(self, x):
        identity = x.clone()
        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):
                # Bernoulli is one --> alive
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                out += identity
            else:
                self.conv1.weight.requires_grad = False
                self.conv1.weight.requires_grad = False
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                
                out = identity
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity
        out = self.relu(out)
        return out
    
class StoDepth_Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_Bottleneck, self).__init_()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag
    
    def forward(self, x):
        identity = x.clone()
        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True
                self.conv3.weight.requires_grad = True
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)
                
                out = self.conv3(out)
                out = self.bn3(out)
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                
                out += identity
            else:
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                self.conv3.weight.requires_grad = False
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = identity
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity
        out = self.relu(out)
        return out
    
class ResNet_StopDepth_lineardecay(nn.Module):
    def __init__(self, block, prob_0_L, multFlag, layers, num_classes=10, zero_init_residual=False):
        super(ResNet_StopDepth_lineardecay, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        
        # linear-decay SD
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(layers)-1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1])
        self.layer3 = self._make_layer(block, 64, layers[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNet_StopDepth_lineardecay):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNet_StopDepth_lineardecay):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplaces != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes*block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes, stride, downsample))
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




















