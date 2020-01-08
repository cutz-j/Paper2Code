# Copyright 
# https://github.com/clovaai/CutMix-PyTorch
# =============================================================================
import os
os.chdir('d:/github/Paper2Code')
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import distributed
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from PyramidNet import model_pytorch as PN
import numpy as np

best_err1 = 100