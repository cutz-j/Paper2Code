import os
import glob
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Activation, Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras import backend as K
from fractalnet import FractalNet

# paper implementation details
NB_CLASSES = 10
NB_EPOCHS = 400
LEARN_START = 0.02
BATCH_SIZE = 100
MOMENTUM = 0.9
Dropout = [0., 0.1, 0.2, 0.3, 0.4]
CONV = [(3, 3, 64), (3, 3, 128), (3, 3, 256), (3, 3, 512), (2, 2, 512)]

# cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

Y_train = to_categorical(y_train, NB_CLASSES)
Y_test = to_categorical(y_test, NB_CLASSES)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.
X_test /= 255.

def learning_rate(epoch):
    if epoch < 200:
        return 0.02
    if epoch < 300:
        return 0.002
    if epoch < 350:
        return 0.0002
    if epoch < 375:
        return 0.00002
    return 0.000002

# build network
im_in = Input(shape=(32, 32, 3))
output = FractalNet(B=5, C=3, conv=CONV, drop_path=0.15, dropout=Dropout, deepest=False)(im_in)
output = Flatten()(output)
output = Dense(NB_CLASSES, init='glorot_normal')(output)
output = Activation('softmax')(output)
model = Model(im_in, output)
optimizer = SGD(lr=LEARN_START, momentum=MOMENTUM, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
plot_model(model, to_file='model.png', show_shapes=True)

# train
learn = LearningRateScheduler(learning_rate)
model.fit(x=X_train, y=Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS,
          validation_data=(X_test, Y_test), callbacks=[learn])
