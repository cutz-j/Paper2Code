import os
os.chdir("D:/github/Paper2Code")
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from SqueezeExcitationNet.senet import SENet
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau

BATCH_SIZE = 1024
NUM_CLASSES = 10
epochs = 100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

sn = SENet(input_shape=(32, 32, 3), classes=10, reduction_ratio=16)
sn.summary()

#dn.compile(optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True), loss='categorical_crossentropy', 
#           metrics=['acc'])
sn.compile(optimizer=SGD(lr=0.1), loss='categorical_crossentropy', metrics=['acc'])


callback_list = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)]

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
datagen.fit(x_train)
sn.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), 
                 epochs=epochs, callbacks=callback_list)