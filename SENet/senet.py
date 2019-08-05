from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape, preprocess_input
from keras.layers import Activation, Conv2D, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D
from keras.layers import Input, Dense, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.layers import DepthwiseConv2D, Dropout, concatenate, Reshape, Multiply
from keras.models import Model
from keras.regularizers import l2
import numpy as np

def SENet(input_shape=None, reduction_ratio=16, include_top=True, weights=None, classes=1000,
          **kwargs):
    '''
    Squeeze & Excitation Net for Keras
    SE-Res50Net
    
    input:
    - input_shape:
        image input shape
    - include_top:
        final classification layer
    - reduction:
        float, reduce the number of feature-maps at transition layers
    - weights:
        pre-trained weight
    - classes:
        output classes
    '''
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    img_input = Input(shape=input_shape)
    # 1 layer
    x = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', kernel_initializer='he_normal', use_bias=False, name='conv1')(img_input)
    x = BatchNormalization(epsilon=0.0001, name='conv1_bn')(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same',name='maxpool1')(x)
    
    # res stage 1 (64 64 256) x 3
    x = res_block(x, input_dim=64, out_dim=256, shortcut_conv=True, strided_conv=False, stage=1, block=1)
    x = se_block(x, out_dim=256, reduction_ratio=reduction_ratio, layer_name='se_block_%d_%d'%(1, 1))
    for i in range(2):
        x = res_block(x, input_dim=64, out_dim=256, shortcut_conv=False, strided_conv=False, stage=1, block=i+2)
        x = se_block(x, out_dim=256, reduction_ratio=reduction_ratio, layer_name='se_block_%d_%d'%(1, i+2))
    
    # res stage 2 (128 128 512) x 4
    x = res_block(x, input_dim=128, out_dim=512, shortcut_conv=True, strided_conv=True, stage=2, block=1)
    x = se_block(x, out_dim=512, reduction_ratio=reduction_ratio, layer_name='se_block_%d_%d'%(2,1))
    for i in range(3):
        x = res_block(x, input_dim=64, out_dim=512, shortcut_conv=False, strided_conv=False, stage=2, block=(i+2))
        x = se_block(x, out_dim=512, reduction_ratio=reduction_ratio, layer_name='se_block_%d_%d'%(2,i+2))
    
    # res stage 3 (256 256 1024) x 6
    x = res_block(x, input_dim=256, out_dim=1024, shortcut_conv=True, strided_conv=True, stage=3, block=1)
    x = se_block(x, out_dim=1024, reduction_ratio=reduction_ratio, layer_name='se_block_%d_%d'%(3,1))
    for i in range(5):
        x = res_block(x, input_dim=256, out_dim=1024, shortcut_conv=False, strided_conv=False, stage=3, block=(i+2))
        x = se_block(x, out_dim=1024, reduction_ratio=reduction_ratio, layer_name='se_block_%d_%d'%(3,i+2))
    
    # res stage 4 (512 512 2048) x 3
    x = res_block(x, input_dim=512, out_dim=2048, shortcut_conv=True, strided_conv=True,stage=4, block=1)
    x = se_block(x, out_dim=2048, reduction_ratio=reduction_ratio, layer_name='se_block_%d_%d'%(4,1))
    for i in range(2):
        x = res_block(x, input_dim=512, out_dim=2048, shortcut_conv=False, strided_conv=False, stage=4, block=(i+2))
        x = se_block(x, out_dim=2048, reduction_ratio=reduction_ratio, layer_name='se_block_%d_%d'%(4,i+2))
    
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation=None, name='dense')(x)
        x = Activation('softmax')(x)
    
    model = Model(img_input, x)
    return model

def se_block(x, out_dim, reduction_ratio=16, layer_name=None):
    '''
    Squeeze Excitation block
    x --> GAP (squeeze) --> FC Dense --> Relu --> FC Dense --> sigmoid --> Scale
      ----------------------- Element by Mul ------------------------------>
    
    input:
        - x:
            input tensor
        - out_dim:
            output dimension
        - reduction_ratio:
            channel reduction
        - layer_name:
            name
    '''
    # squeeze
    squeeze = GlobalAveragePooling2D()(x)
    
    # excitation
    excitation = Dense(units=int(out_dim/reduction_ratio), activation=None, kernel_initializer='he_normal',name=layer_name+'_1')(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim, activation=None,kernel_initializer='he_normal',name=layer_name+'_2')(excitation)
    excitation = Reshape(target_shape=[1, 1, out_dim])(excitation)
    
    # scale with add (skip connection)
    scale = Multiply()([x, excitation])
    return scale
        
def res_block(x, input_dim, out_dim, eps=0.0001, shortcut_conv=False, strided_conv=True, stage=None, block=None):
    '''
    residual block for resnet-50
    x --> conv 1x1 (input_dim) --> conv 3x3 (input_dim) --> conv 1x1 (out_dim) --> Add
      --------------------------------------------------------------------------->
    
    input:
        - x:
            input tensor
        - input_dim:
            channel dimension reduction
        - out_dim:
            output dim
        - eps:
            Batch Normalization Epsilon
        - shortcut_conv:
            shortcut_conv (channel_dim)
        - strided_conv:
            1x1 strided 2 conv (down-sampling)
        - stage:
            resnet stage
        - block:
            block num in stage (iter)
    '''
    shortcut = x
    
    if shortcut_conv:
        shortcut = Conv2D(filters=out_dim, kernel_size=(1, 1), strides=2 if strided_conv==True else 1, padding='same',
                          kernel_initializer='he_normal', name='conv_shortcut_%d_%d'%(stage,block))(shortcut)
        shortcut = BatchNormalization(epsilon=eps, name='bn_shortcut_%d_%d'%(stage,block))(shortcut)
    
    x = Conv2D(filters=input_dim, kernel_size=(1, 1), strides=2 if strided_conv==True else 1, padding='same',
               kernel_initializer='he_normal', name='res_conv1_%d_%d'%(stage, block))(x)
    x = BatchNormalization(epsilon=eps, name='res_bn1_%d_%d'%(stage, block))(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=input_dim, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', name='res_conv2_%d_%d'%(stage, block))(x)
    x = BatchNormalization(epsilon=eps, name='res_bn2_%d_%d'%(stage, block))(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=out_dim, kernel_size=(1, 1), strides=1, padding='same',
               kernel_initializer='he_normal', name='res_conv3_%d_%d'%(stage, block))(x)
    x = BatchNormalization(epsilon=eps, name='res_bn3_%d_%d'%(stage, block))(x)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x