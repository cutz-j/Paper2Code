from keras import layers
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

def inception_module(x, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32):
    '''
    function: inception module
    '''
    x_1 = layers.Conv2D(o_1, 1, padding='same')(x)
    
    x_2 = layers.Conv2D(r_3, 1, padding='same')(x)
    x_2 = layers.Conv2D(o_3, 3, padding='same')(x_2)
    
    x_3 = layers.Conv2D(r_5, 1, padding='same')(x)
    x_3 = layers.Conv2D(o_5, 5, padding='same')(x_3)
    
    x_4 = layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    x_4 = layers.Conv2D(pool, 1, padding='same')(x_4)
    
    return layers.concatenate([x_1, x_2, x_3, x_4])


input_shape = (224, 224, 3)

input_data = layers.Input(shape=input_shape)
x = layers.Conv2D(units=64, kernel=7, strides=2, padding='same')(input_data)
x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 1, strides=1)(x)
x = layers.Conv2D(192, 3, strides=1, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = inception_module(x, o_1=64, r_3=96, o_3=128, r_5=16, o_5=32, pool=32)
x = inception_module(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = layers.