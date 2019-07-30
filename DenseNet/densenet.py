from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape, preprocess_input
from keras.layers import Activation, Conv2D, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D
from keras.layers import Input, Dense, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.layers import DepthwiseConv2D, Reshape
from keras.models import Model
import numpy as np


def DenseNet(input_shape=None, layers_num=50, include_top=True, weights=None, classes=1000, **kwargs):
    '''
    ResNEXT implementation for Keras
    paper uses block c. (3x3 group convolution by cardinality number.)
    ResNext has 50 and 101 layers
    
    input:
        - input_shape:
            image input shape
        - include_top:
            final classification layer
        - scale_factor:
            scales the number of "output channels"
        - weights:
            pre-trained weight
        - classes:
            output classes
        - groups:
            group convolution dividing channels  
    
    '''
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    
    img_input = Input(shape=input_shape)
    
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', 
               use_bias=False, name='conv1_conv')(img_input)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool1_pool')(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    