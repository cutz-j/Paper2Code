from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape, preprocess_input
from keras.layers import Activation, Conv2D, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D
from keras.layers import Input, Dense, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.layers import DepthwiseConv2D, Reshape, Dropout, concatenate
from keras.models import Model
from keras.regularizers import l2
import numpy as np
#import pandas as pd


def DenseNet(input_shape=None, layers_num=121, reduction=0.0, include_top=True, weights=None, classes=1000,
             dropout_rate=0.2, **kwargs):
    '''
    DenseNet implementation for Keras
    paper uses block c. (3x3 group convolution by cardinality number.)
    DenseNet has 121, 169, 201 and 264 layers
    
    input:
        - input_shape:
            image input shape
        - include_top:
            final classification layer
        - layers_num:
            DenseNet has 121, 169, 201 and 264 layers
        - growth_rate:
            dense block channels
        - reduction:
            float, reduce the number of feature-maps at transition layers
        - weights:
            pre-trained weight
        - classes:
            output classes 
    
    '''
    
    layer_iter = {121: {'num': [6, 12, 24, 16], 'filter':64, 'growth_rate':32},
                  161: {'num': [6, 12, 36, 24], 'filter':96, 'growth_rate':48},
                  169: {'num': [6, 12, 32, 32], 'filter':64, 'growth_rate':32},
                  201: {'num': [6, 12, 48, 32], 'filter':64, 'growth_rate':32},
                  264: {'num': [6, 12, 64, 48], 'filter':64, 'growth_rate':32}}
    layers_num_list = layer_iter[layers_num]['num']
    nb_filter = layer_iter[layers_num]['filter']
    growth_rate = layer_iter[layers_num]['growth_rate']
    compression_ratio = 1.0 - reduction
    eps = 1.1e-5
    
    if layers_num not in layer_iter:
        assert "choose [121, 169, 201, 264]"
    
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    
    img_input = Input(shape=input_shape)
    
    # first layer #
    x = Conv2D(filters=nb_filter, kernel_size=(7, 7), strides=2, padding='same', 
               use_bias=False, kernel_initializer='he_normal', name='conv1_conv')(img_input)
    x = BatchNormalization(epsilon=eps, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool1_pool')(x)
    
    for i in range(3):
        x, nb_filter = dense_block(x, layers_num_list[i], nb_filter=nb_filter, growth_rate=growth_rate,
                                    bottleneck=True, name='dense_block_%d'%(i+1))
        # transition layers #
        nb_filter = int(nb_filter * compression_ratio)
        x = BatchNormalization(epsilon=1.1e-5, name='transition_bn_%d'%(i+1))(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=nb_filter, kernel_size=(1, 1), strides=1, padding='same',
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
                   name='transition_conv_%d'%(i+1))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='transition_pool_%d'%(i+1))(x)
    
    x, nb_filter = dense_block(x, layers_num_list[3], nb_filter=nb_filter, growth_rate=growth_rate,
                                bottleneck=True, name='dense_block_4')
    
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation=None, name='Dense')(x)
        x = Activation('softmax', name='softmax')(x)
        
    model = Model(img_input, x, name='model_name')
    return model

def conv_block(x, nb_filter, dropout_rate=None, weight_decay=1e-4, name=None):
    '''
    conv Block
    BN -> Relu -> 1x1 Conv -> BN -> Relu -> 3x3 Conv
    
    input:
        - x:
            tensor
        - nb_filters:
            input channel dimension
        - weights_decay:
            initialized weight decay    \
        - dropout_rate:
            dropout_rate = 0.2
    '''  
    x = BatchNormalization(epsilon=1.1e-5, name=name+'_bn')(x)
    x = Activation('relu')(x)
    
    inter_channel = nb_filter * 4
    x = Conv2D(inter_channel, kernel_size=(1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               name=name+'_bottleneck_conv')(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    
    x = BatchNormalization(epsilon=1e-4, name=name+'_bottleneck_bn')(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, kernel_size=(3, 3), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               name=name+'_conv')(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    return x


    
def dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck, weight_decay=1e-4, dropout_rate=0.2, name=None):
    '''
    Dense Block
    cocnatenated connection from all preceding layers
    
    input:
        - x:
            tensor
        - nb_layers:
            iteration for in-block
        - nb_filters:
            input channel dimension
        - growth_rate:
            dense growth block channels
        - bottleneck:
            1x1 compression ratio
        - weights_decay:
            initialized weight decay    \
        - dropout_rate:
            dropout_rate = 0.2
    '''
    for i in range(nb_layers):
        cb = conv_block(x, nb_filter=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay,
                        name=name+'block_%i'%(i+1))
        x = concatenate([x, cb])
        
        nb_filter += growth_rate
    
    return x, nb_filter
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    