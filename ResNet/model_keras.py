##############################################################################
# ResNet v2: Identity Mappings in Deep Residual Networks
# https://arxiv.org/pdf/1603.05027.pdf
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
##############################################################################
from keras import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.datasets import cifar10
import numpy as np
import os

##########################################
# ResNet parameters
# depth = Layer Length
##########################################
n = 111
depth = n * 9 + 2 

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=False):
    """
    # Fucntion: resnet_layer
    # Arguments:
        inputs: input layer
        num_filters: output_filters num
        kernel_size: kernel_size
        strides
        activation
        batch_normalization: boolean
        conv_first: conv-bn-actfn(v1) / bn-actfn-conv(v2)
    # Returns:
        x (tensor): tensor
    """
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                  kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:  # V1 (conv-bn-actfn)
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = conv(x)
    return x
        

def ResNet_v2(input_shape, depth, num_classes=10):
    """
    # Function: ResNet V2
        (1 x 1) - (3 x3) - (1 x 1)
        BN - ReLU - Conv2D - BN - ReLU - Conv - ADD
        conv1 :  32x32, 16
        stage 0: 32x32, 64
        stage 1: 16x16, 128
        stage 2:  8x8,  256
    # Arguments:
        input_shape: (32x32x3)
        depth: num of conv layers
        num_classes: 10
    """
    num_filters_in = 16
    num_res_blocks = int((depth-2) / 9)
    inputs = Input(shape=input_shape)
    
    # frist 
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)
    
    for stage in range(3):
        # 3 stage
        for res_block in range(num_res_blocks):
            activation = 'relu'
            strides = 1
            # Stage 0
            if stage == 0:
                num_filters_out = num_filters_in * 4 # 64
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
            # stage 1, 2 first layer
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2
            
            # resdidual unit
            y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides, activation=activation,
                             batch_normalization=batch_normalization, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False)
    
            if res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            x = Add()([x, y])
        
        num_filters_in = num_filters_out
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, kernel_initializer='he_normal')(x)
    outputs = Activation('softmax')(outputs)
    model = Model(inputs, outputs)
    return model