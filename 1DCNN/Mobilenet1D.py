import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, MaxPooling1D, GlobalAveragePooling1D, Dense, add
from keras.layers import Dropout, Lambda, Bidirectional, Flatten, BatchNormalization, Activation, SeparableConv1D
from keras.layers import ZeroPadding1D, ReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tqdm import tqdm, trange
from keras import backend as K
from keras.optimizers import Adam


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def mobilenet1D(input_shape, alpha=1.0, **kwargs):
    
    # model parameter #
    img_input = Input(shape=input_shape)
    channel_axis = -1
    first_block_filters = _make_divisible(32 * alpha, 8)
    
    x = Conv1D(first_block_filters, kernel_size=3, strides=1, padding='same', use_bias=False,name='Conv1')(img_input)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = ReLU(max_value=6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=1, block_id=1)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=2)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=3)
    
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=2, expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=1, expansion=6, block_id=5)
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=1, expansion=6, block_id=6)
    
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=2, expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1, expansion=6, block_id=8)    
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1, expansion=6, block_id=9)
    
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=768, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=768, alpha=alpha, stride=1, expansion=6, block_id=12)
    
#     x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
#     x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)
#     x = _inverted_res_block(x, filters=1024, alpha=alpha, stride=2, expansion=6, block_id=8)
#     x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=1024, block_id=9)
#     x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)
    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1024

    x = Conv1D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(axis=channel_axis,epsilon=1e-3,momentum=0.999,name='Conv_1_bn')(x)
    x = ReLU(6., name='out_relu')(x)


    # Create model.
    model = Model(img_input, x)
    return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = -1
    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = Conv1D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
#    if stride == 2:
#        x = ZeroPadding1D(padding=1, name=prefix + 'pad')(x)
    x = SeparableConv1D(in_channels, kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same', name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv1D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3,momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return add([inputs, x])
    return x