
import numpy as np

def StochasticDepth(input_shape=None, include_top=True, weights=None, classes=1000, **kwargs):
    '''
    Stochastic Depth implementation for Pytorch
    
    input:
    - input_shape:
        image input shape
    - include_top:
        final classification layer
    - expected depth:
        float, reduce the number of feature-maps at transition layers
    - weights:
        pre-trained weight
    - classes:
        output classes
    '''