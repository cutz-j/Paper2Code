import os
from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape, preprocess_input
from keras.layers import Activation, Conv2D, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Input, Dense, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda, Cropping2D
from keras.layers import DepthwiseConv2D, Reshape, Dropout, concatenate, ZeroPadding2D, SeparableConv2D, add
from keras.models import Model
from keras.regularizers import l2

def NASNet(input_shape=None, penultimate_filters=768, num_blocks=6, skip_reduction=True,
           skip_reduction_layer_input=True, filter_multiplier=2,  stem_filters=32,
           dropout=0.5, weight_decay=5e-4,  include_top=True, weights=None, input_tensor=None, 
           pooling=None, classes=10, default_size=32, **kwargs):
    '''
    function: NASNet-A cifar-10 implementation for keras
    inputs:
        - input_shape: (32, 32, 3)
        - penultimate_filters: Number of filters in the penultimate layer.
            NASNet models use (N @ P)
                N is the numbeer of blocks
                P is the number of penultimate filters
        - num_blocks: the number of blocks iteration
        - skip_reduction: Wheter to skip the reduction step at the tail end of net
        - filter_multiplier: Width(channel) of network
        - include_top: include classification layer
        - weights
        - input_tensor
        - pooling: If include_top is False, kinds of pooling layer
        - classes
        - default_size
    '''
    global BN_DECAY, BN_EPS
    BN_DECAY = 0.9
    BN_EPS = 1e-5
    
    input_shape = _obtain_input_shape(input_shape, default_size=default_size,
                                      min_size=32, data_format=K.image_data_format(),
                                      require_flatten=True, weights=weights)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    if penultimate_filters % (24 * (filter_multiplier ** 2)) != 0:
        raise ValueError('For NASNet-A models, the `penultimate_filters` must be a multiple '
            'of 24 * (`filter_multiplier` ** 2). Current value: %d' % penultimate_filters)
    
    channel_dim = 1 if K.image_data_format() == 'channel_first' else -1
    filters = penultimate_filters // 24
    
    # Stem
    x = Conv2D(stem_filters, (3, 3), strides=1, padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=channel_dim, momentum=BN_DECAY, epsilon=BN_EPS, name='stem_bn1')(x)
    p = None
    
    # Normal Cell x N
    for i in range(num_blocks):
        x, p = _normal_A(x, p, filters, weight_decay, block_id='%d'%(i))
    
    # Reduction Cell
    x, p0 = _reduction_A(x, p, filters*filter_multiplier, weight_decay, block_id='%d'%(num_blocks))
    
    # Normal Cell x N
    for i in range(num_blocks):
        x, p = _normal_A(x, p, filters*filter_multiplier, weight_decay, block_id='%d' %(num_blocks+i+1))
    
    # Reduction Cell
    x, p0 = _reduction_A(x, p, filters*filter_multiplier**2, weight_decay, block_id='reduce_%d'%(2*num_blocks))
    
    # Normal Cell x N
    for i in range(num_blocks):
        x, p = _normal_A(x, p, filters*filter_multiplier**2, weight_decay, block_id='%d' %(2*num_blocks+i+1))
    
    x = Activation('relu')(x)
    
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='softmax')(x)
    else:
        if pooling=='avg':
            x = GlobalAveragePooling2D()(x)
        else:
            x = GlobalMaxPooling2D()(x)
            
    inputs = img_input
    model = Model(inputs, x, name='NASNet')
    
    if weights:
        model.load_weights(weights)
    
    return model
    
def _separable_conv_block(ip, filters, kernel_size=(3, 3), strides=1, weight_decay=5e-5, block_id=None):
    '''
    function: Adds 2 blocks of [relu-separable_conv-BN]
    inputs:
        - ip: input_tensor
        - filters: Number of output filter
        - kernel_size: sp_conv kernel size
        - strides: Strided Conv
        - block_id: block
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    with K.name_scope('separable_conv_block_%s' % block_id):
        x = Activation('relu')(ip)
        x = SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False,
                            depthwise_initializer='he_normal', pointwise_initializer='he_normal',
                            depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                            name='separable_conv_1_%s' % block_id)(x)
        x = BatchNormalization(axis=channel_dim, momentum=BN_DECAY, epsilon=BN_EPS,
                               name='separable_conv_1_bn_%s' % (block_id))(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, kernel_size, padding='same', use_bias=False,
                            depthwise_initializer='he_normal', pointwise_initializer='he_normal',
                            depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                            name='separable_conv_2_%s' % block_id)(x)
        x = BatchNormalization(axis=channel_dim, momentum=BN_DECAY, epsilon=BN_EPS,
                               name='separable_conv_2_bn_%s' % (block_id))(x)
    return x
    
def _adjust_block(p, ip, filters, weights_decay=5e-5, block_id=None):
    '''
    # Functions: Adjusts the input 'p' to match the shape of the 'input'
    # Arguments:
        p: input tensor
        ip: input tensor matched
        filters: number of output filters
        weight_decay: l2 reg
        id: string
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    img_dim = 2 if K.image_data_format() == 'channels_first' else -2
    with K.name_scope('adjust_block'):
        if p == None:
            p = ip
        
        elif p._keras_shape[img_dim] != ip._keras_shape[img_dim]:
            with K.name_scope('adjust_reduction_block_%s' % block_id):
                p = Activation('relu', name='adjust_relu_1_%s' % block_id)(p)
                
                p1 = AveragePooling2D((1, 1), strides=2, padding='valid', name='adjust_avg_pool_1_%s'%block_id)(p)
                p1 = Conv2D(filters//2, (1, 1), padding='same', use_bias=False,
                            kernel_regularizer=l2(weights_decay), kernel_initializer='he_normal',
                            name='adjust_conv_1_%s' % block_id)(p1)
                
                p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = Cropping2D(cropping=((1,0), (1,0)))(p2)
                p2 = AveragePooling2D((1, 1), strides=2, padding='valid', name='adjust_avg_pool_2_%s' %block_id)(p2)
                p2 = Conv2D(filters//2, (1, 1), padding='same', use_bias=False,
                            kernel_regularizer=l2(weights_decay), kernel_initializer='he_normal',
                            name='adjust_conv_2_%s'%block_id)(p2)
                p = concatenate([p1, p2], axis=channel_dim)
                p = BatchNormalization(axis=channel_dim, momentum=BN_DECAY, epsilon=BN_EPS,
                                       name='adjust_bn_%s'%block_id)(p)
        elif p._keras_shape[channel_dim] != filters:
            with K.name_scope('adjust_projection_block_%s'%block_id):
                p = Activation('relu')(p)
                p = Conv2D(filters, (1, 1), strides=1, padding='same', use_bias=False,
                            kernel_regularizer=l2(weights_decay), kernel_initializer='he_normal',
                            name='adjust_conv_projection_%s'%block_id)(p)
                p = BatchNormalization(axis=channel_dim, momentum=BN_DECAY, epsilon=BN_EPS,
                                       name='adjust_bn_%s'%block_id)(p)
    return p

def _normal_A(ip, p, filters, weight_decay=5e-5, block_id=None):
    """
    # Function: Normal cell for NASNet A
    # Arguments:
        ip: input tensor
        p: input tensor
        filters: output filters
        weight_decay: l2
        id: string
    """
    channel_dim = 1 if K.image_data_format() == 'channels_firts' else -1
    with K.name_scope('normal_A_block_%s' %block_id):
        p = _adjust_block(p, ip, filters, weight_decay, block_id)
        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=1, padding='same', use_bias=False,
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
        h = BatchNormalization(axis=channel_dim, momentum=BN_DECAY, epsilon=BN_EPS,
                               name='normal_bn_1_%s'%block_id)(h)
        
        with K.name_scope('block_1'):
            x1 = _separable_conv_block(h, filters, weight_decay=weight_decay,
                                       block_id='normal_left1_%s'%block_id)
            x1 = add([x1, h])
        with K.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, weight_decay=weight_decay,
                                         block_id='normal_left2_%s'%block_id)
            x2_2 = _separable_conv_block(h, filters, kernel_size=(5, 5), weight_decay=weight_decay,
                                         block_id='normal_right2_%s'%block_id)
            x2 = add([x2_1, x2_2])
        with K.name_scope('block_3'):
            x3 = AveragePooling2D((3, 3), strides=1, padding='same', 
                                    name='normal_left3_%s'%block_id)(h)
            x3 = add([x3, p])
        with K.name_scope('block_4'):
            x4_1 = AveragePooling2D((3, 3), strides=1, padding='same', name='normal_left4_%s'%(block_id))(p)
            x4_2 = AveragePooling2D((3, 3), strides=1, padding='same', name='normal_right4_%s'%(block_id))(p)
            x4 = add([x4_1, x4_2])
        with K.name_scope('block_5'):
            x5_1 = _separable_conv_block(p, filters, kernel_size=(5, 5), weight_decay=weight_decay,
                                         block_id='normal_left5_%s'%block_id)
            x5_2 = _separable_conv_block(p, filters, kernel_size=(3, 3), weight_decay=weight_decay,
                                         block_id='normal_right5_%s'%block_id)
            x5 = add([x5_1, x5_2])
        x = concatenate([p, x1, x2, x3, x4, x5], axis=channel_dim, name='normal_concat_%s'%block_id)
    return x, ip

def _reduction_A(ip, p, filters, weight_decay=5e-5, block_id=None):
    """
    # Function: Size reduction cell for NASNet-A
    # Arguments:
        ip: input tensor x
        p: input tensor p
        filters: number of output filters
        weight_decay: l2 reg
        block_id
    """
    channel_dim = 1 if K.image_data_format() == 'channels_firts' else -1
    with K.name_scope('reduction_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, weight_decay, block_id)
        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=1, padding='same', use_bias=False,
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
        h = BatchNormalization(axis=channel_dim, momentum=BN_DECAY, epsilon=BN_EPS)(h)
        
        # 우측 부터
        with K.name_scope('block_5'):
            x5_1 = AveragePooling2D((3, 3), strides=2, padding='same', name='reduction_left5_%s'%block_id)(h)
            x5_2 = _separable_conv_block(p, filters, (5, 5), strides=2, weight_decay=weight_decay,
                                         block_id='reduction_right5_%s'%block_id)
            x5 = add([x5_1, x5_2])
        with K.name_scope('block_4'):
            x4_1 = MaxPooling2D((3, 3), strides=2, padding='same', name='reduction_left4_%s'%block_id)(h)
            x4_2 = _separable_conv_block(p, filters, (7, 7), strides=2, weight_decay=weight_decay,
                                         block_id='reduction_right4_%s'%block_id)
            x4 = add([x4_1, x4_2])
        with K.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, (7, 7), strides=2, weight_decay=weight_decay,
                                         block_id='reduction_left2_%s'%block_id)
            x2_2 = _separable_conv_block(h, filters, (5, 5), strides=2, weight_decay=weight_decay,
                                         block_id='reduction_right2_%s'%block_id)
            x2 = add([x2_1, x2_2])
        with K.name_scope('block_3'):
            x3_1 = AveragePooling2D((3, 3), strides=1, padding='same', name='reduction_left3_%s'%block_id)(x2)
            x3 = add([x3_1, x4])
        with K.name_scope('block_1'):
            x1_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                                name='reduction_left1_%s'%(block_id))(h)
            x1_2 = _separable_conv_block(x2, filters, (3, 3), weight_decay=weight_decay,
                                         block_id='reduction_right1_%s'%block_id)
            x1 = add([x1_1, x1_2])
        x = concatenate([x1, x3, x4, x5], axis=channel_dim, name='reudcton_concat_%s'%block_id)
        return x, ip  
    
def correct_pad(K, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            