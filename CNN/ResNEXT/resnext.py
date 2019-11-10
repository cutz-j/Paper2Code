from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape, preprocess_input
from keras.layers import Activation, Conv2D, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D
from keras.layers import Input, Dense, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.layers import DepthwiseConv2D, Reshape
from keras.models import Model
import numpy as np


def ResNEXT(input_shape=None, cardinality=32, layers_num=50, include_top=True, weights=None, classes=1000, **kwargs):
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
    x = BatchNormalization(epsilon=1.001e-5, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool1_pool')(x)
    
    # conv2 (x3) #
    x = block_c(x, filters=128, kernel_size=3, strides=2, cardinality=cardinality, conv_shortcut=True, name='conv2_block1')
    for i in range(2):
        x = block_c(x, filters=128, kernel_size=3, strides=1, cardinality=cardinality, name='conv2_block%d' %(i+2))
    
    # conv3 (x4) #
    x = block_c(x, filters=256, kernel_size=3, strides=2, cardinality=cardinality, conv_shortcut=True, name='conv3_block1')
    for i in range(3):
        x = block_c(x, filters=256, kernel_size=3, strides=1, cardinality=cardinality, name='conv3_block%d' %(i+2))
    
    # conv4 (50: x6, 101: x23) #
    x = block_c(x, filters=512, kernel_size=3, strides=2, cardinality=cardinality, conv_shortcut=True, name='conv4_block1')
    for i in range(5 if layers_num == 50 else 22):
        x = block_c(x, filters=512, kernel_size=3, strides=1, cardinality=cardinality, name='conv4_block%d' %(i+2))
    
    # conv5 (x3) #
    x = block_c(x, filters=1024, kernel_size=3, strides=2, cardinality=cardinality, conv_shortcut=True, name='conv5_block1')
    for i in range(2):
        x = block_c(x, filters=1024, kernel_size=3, strides=1, cardinality=cardinality, name='conv5_block%d' %(i+2))
    
    
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation=None, name='Dense')(x)
        x = Activation('softmax', name='softmax')(x)
    
    model = Model(img_input, x, name='model_name')
    return model


def preprocess_input_keras(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    """
    return preprocess_input(x, mode='torch', **kwargs)

def block_c(x, filters, kernel_size=3, strides=1, cardinality=32, conv_shortcut=False, name=None):
    """
    residual block c means depthwise group conv
    input -> 1x1 -> 3x3 (group conv) -> 1x1 -> add 
       ---------------------------------------->
    -----
    Arguments:
        - x:
            input tensor
        - filters:
            conv channels
        - kernel_size:
            depthwise group conv filters
        - strides:
            stride
        - cardinality:
            group conv num
        - name
    Returns:
        - layer
    """
    
    if conv_shortcut:
        shortcut = Conv2D(filters=(64//cardinality)*filters, kernel_size=1, strides=strides,
                          use_bias=False, name=name+'_0_conv')(x)
        shortcut = BatchNormalization(epsilon=1.001e-5, name=name+'_0_bn')(shortcut)
    
    else:    
        shortcut = x
    
    x = Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False, name=name+'_1_conv')(x)
    x = BatchNormalization(epsilon=1.001e-5, name=name+'_1_bn')(x)
    x = Activation('relu', name=name+'_1_relu')(x)
    
    c = filters // cardinality
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', 
                        depth_multiplier=c, use_bias=False, name=name+'_2_conv')(x)
    x_shape = K.int_shape(x)[1:-1] # (w, h)
    x = Reshape(x_shape + (cardinality, c, c))(x)
    output_shape = x_shape + (cardinality, c) if K.backend() == 'theano' else None
    x = Lambda(lambda x: sum([x[:,:,:,:,i] for i in range(c)]), output_shape=output_shape,
               name=name+'_2_reduce')(x)
    x = Reshape(x_shape + (filters,))(x)
    x = BatchNormalization(epsilon=1.001e-5, name=name+'_2_bn')(x)
    x = Activation('relu', name=name+'_2_relu')(x)
    
    x = Conv2D(filters=(64//cardinality)*filters, kernel_size=1, strides=1, padding='same',
               use_bias=False, name=name+'_3_conv')(x)
    x = BatchNormalization(epsilon=1.001e-5, name=name+'_3_bn')(x)
    
    x = Add(name=name+'_add')([x, shortcut])
    x = Activation('relu', name=name+'_out')(x)
    return x
        
    

    
    
    
    