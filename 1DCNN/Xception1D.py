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


def xception1D(input_shape, **kwargs):
    img_input = Input(shape=input_shape)

    x = Conv1D(32, 3, strides=1, padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(64, 1, strides=1, padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    res = Conv1D(128, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=128, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    res = Conv1D(256, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    
    res = Conv1D(512, kernel_size=1, strides=2, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    
    res = Conv1D(728, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    res = Conv1D(728, kernel_size=1, strides=2, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    res = Conv1D(728, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    res = Conv1D(1024, kernel_size=1, strides=2, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=1024, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=1024, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    res = Conv1D(1024, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=1024, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=1024, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    res = Conv1D(1280, kernel_size=1, strides=2, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=1280, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=1280, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    res = Conv1D(1280, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
    res = BatchNormalization()(res)
    
    x = SeparableConv1D(filters=1280, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv1D(filters=1280, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = add([res, x])
    x = BatchNormalization()(x)
    out = Activation('relu')(x)
    
    model = Model(img_input, out)
    return model