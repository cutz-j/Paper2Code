from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape, preprocess_input
from keras.layers import Activation, Conv2D, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D
from keras.layers import Input, Dense, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.layers import DepthwiseConv2D, Reshape, Dropout, concatenate
from keras.models import Model
from keras.regularizers import l2
import numpy as np

img_input = Input(shape=(270, 270, 9))

x = Conv2D(filters=32, kernel_size=(7, 7), strides=2, padding='same')(img_input)
x = MaxPooling2D(pool_size=2, strides=2)(x)
x = Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same')(x)
x = MaxPooling2D(pool_size=2, strides=2)(x)
x = Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same')(x)
x = MaxPooling2D(pool_size=2, strides=2)(x)
x = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same')(x)
x = MaxPooling2D(pool_size=2, strides=2)(x)

model = Model(img_input, x)
print(model.summary())