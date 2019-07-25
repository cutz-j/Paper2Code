from tqdm import tqdm, trange
from keras.optimizers import Adam
from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, SeparableConv1D, AveragePooling1D
from keras.layers import Input, Activation, Dense, BatchNormalization, Lambda, Flatten, Add, Concatenate
from keras.layers import DepthwiseConv2D
from keras.models import Model
import numpy as np


def Shuffle1D(input_shape, scale_factor=1.0, weights=None, classes=40, groups=8, num_shuffle_units=[3, 7, 3], bottleneck_ratio=4, **kwargs):
    '''
    1DShuffleNet implementation for Keras
    
    input:
        - input_shape:
            image input shape
        - include_top:
            final classification layer
        - scale_factor:
            scales the number of "output channels"
        - weights:
            pre-trained weight
        - classes:
            output classes
        - groups:
            group convolution dividing channels  
    
    '''
    img_input = Input(shape=input_shape)

    x = Conv1D(32, 3, strides=1, padding='same', use_bias=False)(img_input) # 15
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(64, 3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = shuffle_unit(x, 64, 384, groups=groups, strides=2, stage=2, block=1, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 384, 384, groups=groups, strides=1, stage=2, block=2, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 384, 384, groups=groups, strides=1, stage=2, block=3, bottleneck_ratio=bottleneck_ratio)
    
    x = shuffle_unit(x, 384, 512, groups=groups, strides=2, stage=3, block=1, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 512, 512, groups=groups, strides=1, stage=3, block=2, bottleneck_ratio=bottleneck_ratio)    
    x = shuffle_unit(x, 512, 512, groups=groups, strides=1, stage=3, block=3, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 512, 512, groups=groups, strides=1, stage=3, block=4, bottleneck_ratio=bottleneck_ratio)

    #======== middle flow ========= #
    x = shuffle_unit(x, 512, 768, groups=groups, strides=2, stage=4, block=1, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 768, 768, groups=groups, strides=1, stage=4, block=2, bottleneck_ratio=bottleneck_ratio)    
    x = shuffle_unit(x, 768, 768, groups=groups, strides=1, stage=4, block=3, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 768, 768, groups=groups, strides=1, stage=4, block=4, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 768, 768, groups=groups, strides=1, stage=4, block=5, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 768, 768, groups=groups, strides=1, stage=4, block=6, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 768, 768, groups=groups, strides=1, stage=4, block=7, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 768, 768, groups=groups, strides=1, stage=4, block=8, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 768, 768, groups=groups, strides=1, stage=4, block=9, bottleneck_ratio=bottleneck_ratio)
    
    # ======== exit flow ========== #
    x = shuffle_unit(x, 768, 1024, groups=groups, strides=2, stage=5, block=1, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 1024, 1024, groups=groups, strides=1, stage=5, block=2, bottleneck_ratio=bottleneck_ratio)    
    x = shuffle_unit(x, 1024, 1024, groups=groups, strides=1, stage=5, block=3, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 1024, 1024, groups=groups, strides=1, stage=5, block=4, bottleneck_ratio=bottleneck_ratio)
    
    x = shuffle_unit(x, 1024, 1536, groups=groups, strides=2, stage=6, block=1, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 1536, 1536, groups=groups, strides=1, stage=6, block=2, bottleneck_ratio=bottleneck_ratio)    
    x = shuffle_unit(x, 1536, 1536, groups=groups, strides=1, stage=6, block=3, bottleneck_ratio=bottleneck_ratio)
    x = shuffle_unit(x, 1536, 1536, groups=groups, strides=1, stage=6, block=4, bottleneck_ratio=bottleneck_ratio)
    model = Model(img_input, x)
    return model


def shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):
    """
    shuffle unit by strides [1, 2]
    strides=1: 1x1 GConv (BN/ReLU) --> Channel Shuffle --> 3x3 DWConv (BN) --> 1x1 GConv (BN) --> add (ReLU)
    strides=2: 1x1 GConv (BN/ReLU) --> Channel Shuffle --> 3x3 DWConv (s=2, BN) --> 1x1 GConv (BN) --> Concat
                ------------------------> 3x3 AVG Pool (s=2) ------------------------------------>
    ----
    inputs: 
        - inputs: 
            input tensor
        - in_channels: 
            in channel number
        - out_channels:
            output channel number
        - groups:
            group number
        - bottleneck_ratio:
            output channel : bottelneck channel
        - strdes:
            1 or 2
        - stage:
            stage number
        - block:
            block number
    """
    prefix = 'stage%d/block%d' %(stage, block)
    
    bottleneck_channels = int(out_channels // bottleneck_ratio)
    x = GroupConv1D(inputs, in_channels, out_channels=bottleneck_channels, groups=groups, name='%s/1x1_gconv_1' % (prefix))
    x = BatchNormalization(name='%s/bn_gconv_1' % (prefix))(x)
    x = Activation('relu', name='%s/relu_gconv_1' % (prefix))(x)
    x = Lambda(channel_shuffle, arguments={'groups':groups}, name='%s/channel_shuffle'%(prefix))(x)
    x = SeparableConv1D(filters=out_channels if strides==1 else out_channels-in_channels, kernel_size=3, strides=strides, padding='same', 
                        name='%s/1x1_dwconv_1' % (prefix))(x)
    x = BatchNormalization(name='%s/bn_dwconv_1' % (prefix))(x)
    
    if strides == 1:
        res = Add(name='%s/add' % (prefix))([x, inputs])
    else:
        avg = AveragePooling1D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % (prefix))(inputs)
        res = Concatenate(name='%s/concat'  % (prefix))([x, avg])
    
    res = Activation('relu', name='%s/relu_out' % (prefix))(res)
    return res

    
def GroupConv1D(x, in_channels, out_channels, groups=1, kernel=1, strides=1, name=''):
    """
    group Convolution 1D
    group=1 means pointwise convolution
    ----
    input:
        - x:
            input tensor
        - in_channels:
            input channels
        - out_channels:
            output channels
        - groups:
            divide channels by group
        - kernel:
            kernel size
        - strdies:
            strides
        - name
    """
    if groups == 1: # Point-wise Conv
        return Conv1D(filters=out_channels, kerenel_size=kernel, padding='same',
                      use_bias=False, strdies=strides, name=name)(x)

    channel_by_groups = in_channels // groups
    groups_list = []
    
    assert out_channels % groups == 0

    for i in range(groups):
        # channel divide #
        group = Lambda(lambda z: z[:,:, int(channel_by_groups * i): int(channel_by_groups * (i + 1))], name='%s/%d_slice'%(name, i))(x)
        # 각각의 filter channel ( int(0.5 + out_channels / groups)) #
        groups_list.append(Conv1D(int(0.5 + out_channels / groups), kernel_size=kernel, padding='same', use_bias=False, strides=strides,
                                  name='%s_/%d' %(name, i))(group))

    return Concatenate(name='%s/concat' % name)(groups_list)

def channel_shuffle(x, groups):
    """
    channel shuffle --> mix independent correalation
    ---
    input:
        - x:
            input tensor
        - groups:
            group number
    """
    w, in_channels = x.shape.as_list()[1:] # 3d tensor (m, w, c)
    channels_per_group = in_channels // groups
    
    x = K.reshape(x, [-1, w, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 3, 2))
    x = K.reshape(x, [-1, w, in_channels])
    return x