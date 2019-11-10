import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Add, Convolution2D
from keras import Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
import json
import time

def get_p_survival(block=0, nb_total_blocks=110, p_survival_end=0.5, mode='linear_decay'):
    return 1 - ((block + 1) / nb_total_blocks) * (1 - p_survival_end)

def zero_pad_channels(x, pad=0):
    pattern = [[0, 0], [0, 0], [0, 0], [pad- pad // 2, pad // 2]]
    return tf.pad(x, pattern)

def stochastic_survival(y, p_survival=1.0):
    survival = K.random_binomial((1,), p=p_survival)
    return K.in_test_phase(tf.constant(p_survival, dtype='float32')*y, survival*y)

class StochasticDepth(object):
    def __init__(self, nb_classes=10):
        
        self.in_classes = nb_classes
        img_rows, img_cols = 32, 32
        img_channels = 3
        blocks_per_groups = 33
        inputs = Input(shape=(img_rows, img_cols, img_channels))
        x = Conv2D(16, kernel_size=(3, 3), strides=1, kernel_initializer='he_normal', padding='same')(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        
        for i in range(0, blocks_per_groups):
            nb_filters = 16
            x = self.res_block(x, nb_filters=nb_filters, block=i, nb_total_blocks=3*blocks_per_groups, strides=1)
            
        for i in range(0, blocks_per_groups):
            nb_filters = 32
            if i == 0:
                strides = 2
            else:
                strides = 1
            x = self.res_block(x, nb_filters=nb_filters, block=blocks_per_groups+i, nb_total_blocks=3*blocks_per_groups, strides=strides)
        for i in range(0, blocks_per_groups):
            nb_filters = 64
            if i == 0:
                strides = 2
            else:
                strides = 1
            x = self.res_block(x, nb_filters=nb_filters, block=2*blocks_per_groups+i, nb_total_blocks=3*blocks_per_groups, strides=strides)
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.in_classes, activation='softmax')(x)
        self.model = Model(input=inputs, output=predictions)
        self.model.summary()
                
    
    def res_block(self, x, nb_filters, block, nb_total_blocks, strides):
        prev_nb_channels = K.int_shape(x)[-1]
        if strides > 1:
            strides = (2, 2)
            shortcut = AveragePooling2D(pool_size=2, strides=strides)(x)
            if nb_filters > prev_nb_channels:
                shortcut = Conv2D(prev_nb_channels, kernel_size=(1, 1), strides=1, padding='valid')(shortcut)
                shortcut = BatchNormalization()(shortcut)
        else:
            strides = (1, 1)
            shortcut = x
        y = Conv2D(nb_filters, kernel_size=(3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(x)
        y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)
        y = Conv2D(nb_filters, kernel_size=(3, 3), strides=1, kernel_initializer='he_normal', padding='same')(y)
        y = BatchNormalization()(y)
        p_survival = get_p_survival(block=block, nb_total_blocks=nb_total_blocks, p_survival_end=0.5, mode='linear_decay')
        y = Lambda(stochastic_survival, arguments={'p_survival':p_survival})(y)
        out= Add()([y, shortcut])
        out = Activation('relu')(out)
        return out