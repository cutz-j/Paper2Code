from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.layers import Activation, Conv2D, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D
from keras.layers import Input, Dense, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.layers import DepthwiseConv2D
from keras.models import Model

def ShuffleNet(input_shape=None, scale_factor=1.0, include_top=True, weights=None, classes=1000,
               groups=1, num_shuffle_units=[3, 7, 3], bottleneck_ratio=4, **kwargs):
    '''
    ShuffleNet implementation for Keras
    
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
    name = "ShuffleNet_%.2gX_g%d_br_%.2g_%s" % (scale_factor, groups, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=28, 
                                      require_flatten=include_top,
                                      data_format=K.image_data.format())
    img_input = Input(shape=input_shape)
    # first layer #
    x = Conv2D(filters=24, kernel_size=(3, 3), padding='same', use_bias=False,
               strides=(2, 2), activation=None, name="conv1")(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, name="maxpool1")(x)
    
    channel_list = _info(groups)
    channel_list *= scale_factor
    # stage 2 #
    x = shuffle_unit(x, channel_list[0], out_channels=channel_list[1], groups=groups,
                     strides=2, stage=2, block=1, bottleneck_ratio=bottleneck_ratio)
    for i in range(num_shuffle_units[0]):
        x = shuffle_unit(x, in_channels=channel_list[1], out_channels=channel_list[1], groups=groups,
                         strides=1, stage=2, block=i+2, bottleneck_ratio=bottleneck_ratio)
    # stage 3 #    
    x = shuffle_unit(x, in_channels=channel_list[1], out_channels=channel_list[2], groups=groups,
                     strides=2, stage=3, block=1, bottleneck_ratio=bottleneck_ratio)
    for i in range(num_shuffle_units[1]):
        x = shuffle_unit(x, in_channels=channel_list[2], out_channels=channel_list[2], groups=groups,
                         strides=1, stage=3, block=i+2, bottleneck_ratio=bottleneck_ratio)
    # stage 4 #
    x = shuffle_unit(x, in_channels=channel_list[2], out_channels=channel_list[3], groups=groups,
                     strides=2, stage=4, block=1, bottleneck_ratio=bottleneck_ratio)
    for i in range(num_shuffle_units[2]):
        x = shuffle_unit(x, in_channels=channel_list[3], out_channels=channel_list[3], groups=groups,
                         strides=1, stage=4, block=i+2, bottleneck_ratio=bottleneck_ratio)
    x = GlobalAveragePooling2D(name="global_pool")(x)
    model = Model(inputs=img_input, outputs=x, name=name)
    
    if weights is not None:
        model.load_weights(weights, by_name=True)
    
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
    
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    x = GroupConv2D(inputs, in_channels, out_channels=bottleneck_channels,
                    groups=groups, bname='%s/1x1_gconv_1' % prefix)
    x = BatchNormalization(x, name='%s/bn_gconv_1' % prefix)(x)
    x = Activation('relu', name='%s/relu_gconv_1' % prefix)(x)
    
    x = Lambda(channel_shuffle, arguments={'groups':groups}, name='%s/channel_shuffle' % prefix)(x)
    
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False,
                        activation=None, strides=strides, name='%s/1x1_dwconv_1' % prefix)(x)
    x = BatchNormalization(name='%s/bn_dwconv_1' % prefix)(x)
    x = GroupConv2D(x, bottleneck_channels, out_channels=out_channels if strides == 1 else out_channels - in_channels,
                    groups=groups, name='%s/1x1_gconv_2' % prefix)
    x = BatchNormalization(name='%s/bn_gconv_2' % prefix)(x)
    
    if strides == 1:
        res = Add(name='%s/add' % prefix)([x, inputs])
    else:
        avg = AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(inputs)
        res = Concatenate(name='%s/concat'  % prefix)([x, avg])
    
    res = Activation('relu', name='%s/relu_out' % prefix)(res)
    return res

def GroupConv2D(x, in_channels, out_channels, groups=1, kernel=1, strides=1, name=''):
    """
    group Convolution 2D
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
        return Conv2D(filters=out_channels, kerenel_size=kernel, padding='same',
                      use_bias=False, strdies=strides, name=name)(x)
    
    channel_by_groups = in_channels // groups
    groups_list = []
    
    assert out_channels % groups == 0
    
    for i in range(groups):
        offset = i * channel_by_groups
        # channel divide #
        groups = Lambda(lambda z: z[:,:,:, offset: offset + channel_by_groups], 
                        name='%s/g%d_slice' %(name, i))(x)
        # 각각의 filter channel ( int(0.5 + out_channels / groups)) #
        groups_list.append(Conv2D(int(0.5 + out_channels / groups), 
                                  kernel_size=kernel, apdding='same',
                                  use_bias=False, strides=strides,
                                  name='%s_/g%d' %(name, i))(x))
    
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
    w, h, in_channels = x.shape.as_list()[1:] # 4d tensor (m, w, h, c)
    channels_per_group = in_channels // groups
    
    x = K.reshape(x, [-1, w, h, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, w, h, in_channels])
    return x
        
def _info(nb_groups):
    return {
        1: [24, 144, 288, 576],
        2: [24, 200, 400, 800],
        3: [24, 240, 480, 960],
        4: [24, 272, 544, 1088],
        8: [24, 384, 768, 1536]
    }[nb_groups]
    