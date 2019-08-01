import os
os.chdir("D:/github/Paper2Code")
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.applications.densenet import DenseNet121
from DenseNet.densenet import DenseNet
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler

BATCH_SIZE = 64
NUM_CLASSES = 10
epochs = 40

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

dn = DenseNet(input_shape=(32, 32, 3), layers=121, classes=10, reduction=0.5)
dn.summary()

#dn.compile(optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True), loss='categorical_crossentropy', 
#           metrics=['acc'])
dn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])

def schedule(epoch):
    if epoch == 20:
        return 0.01
    elif epoch == 30:
        return 0.001
    else:
        return 0.1
    
callback_list = [LearningRateScheduler(schedule=schedule)]

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
datagen.fit(x_train)
dn.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), 
                 steps_per_epoch=300, epochs=epochs, callbacks=callback_list)