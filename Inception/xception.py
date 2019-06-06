import numpy as np
from keras import models, layers, activations, optimizers, losses, Input, callbacks
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
from keras import backend as K
from keras.applications.xception import Xception
from keras.layers import Activation, BatchNormalization, Conv2D, Conv1D, SeparableConv2D

#def separableConv(x, filters=128):
#    '''
#    function: xception module --> depth-wise separable Conv
#    
#    input:
#        - x: conv layer
#        - filters: filter #
#    
#    output:
#        - x: layer
#    '''
#    point_wise = Conv2D(filters=filters, kernel_size=(1, 1), strides=1, use_bias=False)(x)
#    for i in range(filters):
#        depth_wise = Conv1D(filters=1, kernel_size=3, strides=2, use_bias=False,
#                            padding='same')(point_wise[:,:,:,i])
#    x = layers.concatenate(depth_wise)
#    return x


input_layer = Input(shape=(299, 299, 3))
# ========= entry ========= #
# layer 1 #
x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='valid', use_bias=False)(input_layer)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# layer 2 #
x = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# skip layer 1 #
res = Conv2D(filters=128, kernel_size=(1, 1), strides=2, padding='same', use_bias=False)(x)
res = BatchNormalization()(res)

# layer 3 #
x = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
x = BatchNormalization()(x)

# layer 4 #
x = Activation('relu')(x)
x = SeparableConv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', use_bias=False)(x)
x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = layers.add([x, res])

# skip layer 2 #
res = Conv2D(filters=256, kernel_size=(1, 1), strides=2, padding='same', use_bias=False)(x)
res = BatchNormalization()(res)

# layer 5 #
x = Activation('relu')(x)
x = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
x = BatchNormalization()(x)

# layer 6 #
x = Activation('relu')(x)
x = SeparableConv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', use_bias=False)(x)
x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = layers.add([x, res])

# skip layer 3 #
res = Conv2D(filters=728, kernel_size=(1, 1), strides=2, padding='same', use_bias=False)(x)
res = BatchNormalization()(res)

# layer 7 #
x = Activation('relu')(x)
x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
x = BatchNormalization()(x)

# layer 8 #
x = Activation('relu')(x)
x = SeparableConv2D(filters=728, kernel_size=(3,3), strides=1, padding='same', use_bias=False)(x)
x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = layers.add([x, res])

# ======== middle flow ========= #
for i in range(8):
    # layer 9, 10, 11, 12, 13, 14, 15, 16, 17 #
    res = x
    
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)    

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = layers.add([x, res])    

# ======== exit flow ========== #
# skip layer 4 #
res = Conv2D(filters=1024, kernel_size=(1, 1), strides=2, padding='same', use_bias=False)(x)
res = BatchNormalization()(res)

# layer 18 #
x = Activation('relu')(x)
x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
x = BatchNormalization()(x)

# layer 19 #
x = Activation('relu')(x)
x = SeparableConv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = layers.add([x, res])

# layer 20 #
x = SeparableConv2D(filters=1536, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# layer 21 #
x = SeparableConv2D(filters=2048, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = layers.GlobalAveragePooling2D()(x)
output = layers.Dense(units=1000, activation='softmax')(x)

xception = models.Model(input_layer, output)
xception.summary()

### examples ###
xc = Xception(input_shape=(299, 299, 3))
xc.summary()



















