import keras
import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Merge
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.engine import Layer # class 상속을 위한 클래스

##########################################
# FractalNet parameters
# B = Fractal Blocks numbers
# C = C interwinded columns
##########################################

def tensorflow_categorical(count, seed):
    assert count > 0
    arr = [1.] + [.0 for _ in range(count-1)]
    return tf.random_shuffle(arr, seed)

def random_one(count, seed=None):
    """
    Returns a random array [x0, x1, ... xn] where one is 1 and the others
    Ex: [0, 0, 1, 0]
    """
    if seed is None:
        seed = np.random.randint(1, 10e6)
    return tensorflow_categorical(count=count, seed=seed)

class JoinLayer(Layer):
    """
    Function: behave as Merge during testing. during training it will randomly select
              Global / Local DropPath and avg
        Global: Use the random shared tensor to select the paths
        Local: Sample a random tensor to select the paths
    """
    def __init__(self, drop_p, is_global, global_path, force_path, **kwargs):
        self.p = 1. - drop_p
        self.is_global = is_global
        self.global_path = global_path
        self.uses_learning_phase = True
        self.force_path = force_path
        super(JoinLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.average_shape = list(input_shape[0])[1:]

    def _random_arr(self, count, p):
        return K.random_binomial((count,), p=p)
    
    def _arr_with_one(self, count):
        return random_one(count=count)
    
    def _gen_local_drops(self, count, p):
        """
        Function: Local DropPath
        """
        arr = self._random_arr(count, p)
        drops = K.switch(K.any(arr), arr, self._arr_with_one(count))
        return drops
    
    def _gen_global_path(self, count):
        return self.global_path[:count]
    
    def _drop_path(self, inputs):
        count = len(inputs)
        drops = K.switch(self.is_global, self._gen_global_path(count), self._gen_local_drops(count, self.p))
        ave = K.zeros(shape=self.average_shape)
        for i in range(0, count):
            ave += inputs[i] * drops[i]
        sum = K.sum(drops)
        # check 0 not for division by zero
        ave = K.switch(K.not_equal(sum, 0.), ave/sum, ave)
        return ave
      
    def _ave(self, inputs):
        # element-wise-mean
        ave = inputs[0]
        for inp in inputs[1:]:
            ave += inp
        ave /= len(inputs)
        return ave
    
    def call(self, inputs, mask=None):
        if self.force_path:
            output = self._drop_path(inputs)
        else:
            output = K.in_train_phase(self._drop_path(inputs), self._ave(inputs))
        return output
    
    def get_output_shape_for(self, input_shape):
        return input_shape[0]
    
class JoinLayerGen:
    """
    Function: init seeds for both global droppath switch and global droppout path
                seeds will be used to create the random tensors that the children layers
                will use to know if they must use global dropout and which path to take in case
    """
    def __init__(self, width, global_p=0.5, deepest=False):
        self.global_p = global_p
        self.width = width
        self.switch_seed = np.random.randint(1, 10e6)
        self.path_seed = np.random.randint(1, 10e6)
        self.deepest = deepest
        if deepest:
            self.is_global = K.variable(1.)
            self.path_array = K.variable([1.] + [.0 for _ in range(width-1)])
        else:
            self.is_global = self._build_global_switch()
            self.path_array = self._build_global_path_arr()
        
    def _build_global_path_arr(self):
        # path block when global droppath
        return random_one(seed=self.path_seed, count=self.width)
    
    def _build_global_switch(self):
        # randomly sampled tensor
        # global or local droppath
        return K.equal(K.random_binomial((), p=self.global_p, seed=self.switch_seed), 1.)
    
    def get_join_layer(self, drop_p):
        global_switch = self.is_global
        global_path = self.path_array
        return JoinLayer(drop_p=drop_p, is_global=global_switch, global_path=global_path, force_path=self.deepest)
    
def fractal_conv(filters, nb_row, nb_col, dropout=None):
    def f(prev):
        conv = prev
        conv = Convolution2D(filters, nb_row=nb_col, nb_col=nb_col, init='xavier_normal', border_mode='same')(conv)
        if dropout:
            conv = Dropout(dropout)(conv)
        conv = BatchNormalization(mode=0, axis=1)(conv)
        conv = Activation('relu')(conv)
        return conv
    return f
            
            
            
            
            
            
            
            
            
            
            
            
            
    
#def FractalNet(input_shape=None, B=5, C=4, include_top=True, weights=None, classes=10, **kwargs):
#    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        