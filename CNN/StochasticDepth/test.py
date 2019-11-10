import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, AveragePooling2D, BatchNormalization
from keras.layers import merge, Convolution2D
from keras import Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
import json
import time

## os dir ##
import os
os.chdir("d:/github/Paper2Code/")
from StochasticDepth.model_keras import StochasticDepth


nb_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# reorder dimensions for tensorflow
x_train = np.transpose(x_train.astype('float32'), (0, 2, 3, 1))
mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train)
x_train = (x_train - mean) / std
x_test = np.transpose(x_test.astype('float32'), (0, 2, 3, 1))
x_test = (x_test - mean) / std
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = StochasticDepth()
