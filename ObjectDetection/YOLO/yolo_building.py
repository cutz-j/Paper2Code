import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("D:/github/Paper/YOLO")
from pascal import pascal_voc


def add_variable_summary(tf_variable, summary_name):
  with tf.name_scope(summary_name + '_summary'):
    mean = tf.reduce_mean(tf_variable)
    tf.summary.scalar('Mean', mean)
    with tf.name_scope('standard_deviation'):
        standard_deviation = tf.sqrt(tf.reduce_mean(
            tf.square(tf_variable - mean)))
    tf.summary.scalar('StandardDeviation', standard_deviation)
    tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
    tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
    tf.summary.histogram('Histogram', tf_variable)

def poolingLayer(input_layer, pool_size=[2, 2], strides=2, padding='valid'):
    '''
    function: max-pooling 2x2 s=2
    
    inputs:
        - input_layer: input layer
        - pool_size: base [2, 2]
        - strides: base 2
        - padding: base 'valid'
    
    outputs:
        - layer: pooled layer
    '''
    layer = tf.layers.max_pooling2d(inputs=input_layer, pool_size=pool_size, 
                                    strides=strides, padding=padding)
    add_variable_summary(layer, 'pooling')
    return layer

def convLayer(input_layer, filters, kernel_size=3, strides=1, padding='valid', activation=tf.nn.leaky_relu):
    '''
    function: 3x3 filter size conv layer --> actfn: learky_relu
    
    inputs:
        - input_layer
        - filters: W
        - kernel_size: [3, 3]
        - padding: valid
        - activation: leaky_relu
    
    outputs:
        - layers: conv layer
    '''
    kernel_size = [kernel_size, kernel_size]
    strides = (strides, strides)
    layer = tf.layers.conv2d(inputs=input_layer, filters=filters, kernel_size=kernel_size,
                             activation=activation, padding=padding,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    add_variable_summary(layer, 'convolution')
    return layer

def denseLayer(input_layer, units, activation=tf.nn.leaky_relu):
    '''
    function: dense layer --> actfn: leaky_relu
    
    inputs:
        - input_layer
        - units: W
        - activation: leaky_relu
    
    outputs:
        - layers: dense layer
    '''
    layer = tf.layers.dense(inputs=input_layer, units=units,
                             activation=activation,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    add_variable_summary(layer, 'dense')
    return layer

## hyper parameter setting ##
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

num_class = len(classes) # class #
image_size = 448 # 224 x 2
cell_size = 7
boxes_per_cell = 2
output_size = (cell_size * cell_size) * (num_class + boxes_per_cell * 5) # 각 2개씩
sclae = 1.0 * image_size /cell_size
boundary1 = cell_size * cell_size * num_class
boundary2 = boundary1 + cell_size * cell_size * boxes_per_cell

object_scale = 1.0
noobject_scale= 1.0
class_scale = 2.0
coord_scale = 5.0

learning_rate = 0.0001
batch_size = 64
alpha = 0.1

max_iter = 15000
initial_learning_rate = 0.0001
decay_steps = 30000
decay_rate = 0.1 # 증가
staircase = True

## yolo building ##
images = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, 3], name='images') # [m, 448, 448, 3]

# first layers #    
yolo = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1') # 패딩
cv1 = convLayer(yolo, 64, 7, 2)
L1 = poolingLayer(cv1, [2, 2], 2, 'same')    

# second layers
cv2 = convLayer(L1, 192, 3)
L2 = poolingLayer(cv2, [2, 2], 2, 'same')

# thirs layers #
cv3 = convLayer(L2, 128, 1)
cv4 = convLayer(cv3, 256, 3)
cv5 = convLayer(cv4, 256, 1)
cv6 = convLayer(cv5, 512, 3)
L6 = poolingLayer(cv6, [2, 2], 2, 'same')

# fourth layers #
cv7 = convLayer(L6, 256, 1)
cv8 = convLayer(cv7, 512, 3)    
cv9 = convLayer(cv8, 256, 1)
cv10 = convLayer(cv9, 512, 3)    
cv11 = convLayer(cv10, 256, 1)
cv12 = convLayer(cv11, 512, 3)    
cv13 = convLayer(cv12, 256, 1)
cv14 = convLayer(cv13, 512, 3)
cv15 = convLayer(cv14, 1024, 3)
L15 = poolingLayer(cv15, [2, 2], 2)    
    
# fifth layers #
cv16 = convLayer(L15, 512, 1)
cv17 = convLayer(cv16, 1024, 3)
cv18 = convLayer(cv17, 512, 1)
cv19 = convLayer(cv18, 1024, 3)    
cv20 = convLayer(cv19, 1024, 3)
L20 = tf.pad(cv20, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]))
cv21 = convLayer(L20, 1024, 3, 2)

# sixth layers #
cv22 = convLayer(cv21, 1024, 3)
cv23 = convLayer(cv22, 1024, 3)
yolo = tf.transpose(cv23, [0, 3, 1, 2]) # fc transpose
fc1 = tf.layers.flatten(yolo)
fc2 = denseLayer(fc1, 512)
fc3 = denseLayer(fc2, 4096)

dropout_bool = tf.placeholder(tf.bool)
yolo = tf.layers.dropout(inputs=fc3, rate=0.5, training=dropout_bool)
fc4 = denseLayer(yolo, output_size, None)
    
    
    
    
    
    
    
 







   