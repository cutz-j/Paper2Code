from keras.datasets import mnist
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,AveragePooling2D,Reshape
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt 


def preprocess():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train),28,28, 1))
    x_test = np.reshape(x_test, (len(x_test),28,28, 1))
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)

    return (x_train,y_train,x_test,y_test)

def model(x):
    x1 = Conv2D(11,(1,1),padding = 'same')(x)
    x1 = Flatten()(x1)
    x1 = Dense(900,activation = 'relu')(x1)
    x1 = Dense(900,activation = 'relu')(x1)
    x1 = Reshape((30,30,1),input_shape = x1.shape)(x1)
    x1 = Conv2D(11,(1,1),padding = 'same')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(400,activation = 'relu')(x1)
    x1 = Dense(400,activation = 'relu')(x1)
    x1 = Reshape((20,20,1),input_shape = x1.shape)(x1)
    x1 = Conv2D(11,(1,1),padding = 'same')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(100,activation = 'relu')(x1)
    x1 = Dense(100,activation = 'relu')(x1)
    x1 = Reshape((10,10,1),input_shape = x1.shape)(x1)
    x1 = Conv2D(11,(1,1),padding = 'same')(x1)
    x1 = AveragePooling2D((2,2),padding = 'same')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(10)(x1)
    output = Activation('softmax')(x1)
    return output

input_img = Input(shape = (28,28,1))
NiN = Model(input_img,model(input_img))
NiN.compile(optimizer='adam',loss = 'categorical_crossentropy')
x_train,y_train,x_test,y_test = preprocess()
NiN.fit(x_train,y_train,
            epochs=20,
            batch_size=128,
            shuffle=True)
plt.plot(epochs,loss)
plt.show()