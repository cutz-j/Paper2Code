import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.applications import VGG16
import os
import random
import time
import wget
import tarfile
import numpy as np
import cv2

class MaxUnpoolWithArgmax(Layer):
    ''' 
    max pooling --> max unpool: max된 pixel을 switch로 되돌리기
    '''
    def __init__(self, pooling_argmax, stride=[1,2,2,1], **kwargs):
        '''
        # parameter #
        pooling_argmax = pooling시에 max로 추출된 인덱스
        stride = default 2x2 box, tensorflow stride = [1, f, f , 1]
        ## 생성 함수 ##
        '''
        self.pooling_argmax = pooling_argmax
        self.stride = stride
        super(MaxUnpoolWithArgmax, self).__init__(**kwargs) # 최상위 부모함수 load
        
    def build(self, input_shape):
        '''
        ## parameter ##
        input_shape = [224x224]
        
        ## function ##
        layer 부모함수에서 생성
        '''
        super(MaxUnpoolWithArgmax, self).build(input_shape)
        
    def call(self, inputs):
        ''' ? '''
        input_shape = K.cast(K.shape(inputs), dtype='int64')
        output_shape = (input_shape[0], 
                        input_shape[1] * self.stride[1],
                        input_shape[2] * self.stride[2],
                        input_shape[3])
        
        argmax = self.pooling_argmax ## pooling argmax
        one_like_mask = K.ones_like(argmax) # 1 matrix
        batch_range = K.reshape(K.arange(start=0, stop=input_shape[0], dtype='int64'),
                                shape=[input_shape[0], 1, 1, 1])
        b = one_like_mask * batch_range
        y = argmax // (output_shape[2] * output_shape[3])
        x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = K.arange(start=0, stop=output_shape[3], dtype='int64')
        f = one_like_mask * feature_range
        ## transpose ##
        update_size = tf.size(inputs)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, update_size]))
        values = K.reshape(inputs, [update_size])
        return tf.scatter_nd(indices, values, output_shape)
    
    def cmpute_output_shape(self, input_shape):
        '''
        output shape
        '''
        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3])
    
    def get_config(self):
        '''
        ## function ##
        base config 기록, list 형태로 return
        '''
        base_config = super(MaxUnpoolWithArgmax, self).get_config()
        base_config['pooling_argmax'] = self.pooling_argmax
        base_config['stride'] = self.stride
        return base_config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
























