import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.datasets import cifar10
import numpy as np
import os
from model_keras import ResNet_v2

batch_size = 32
epochs = 200
num_classes = 10
n = 111
depth = n * 9 + 2 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                            # divide inputs by std of dataset
                            featurewise_std_normalization=False,
                            # divide each input by its std
                            samplewise_std_normalization=False,
                            # apply ZCA whitening
                            zca_whitening=False,
                            # epsilon for ZCA whitening
                            zca_epsilon=1e-06,
                            # randomly rotate images in the range (deg 0 to 180)
                            rotation_range=0,
                            # randomly shift images horizontally
                            width_shift_range=0.1,
                            # randomly shift images vertically
                            height_shift_range=0.1,
                            # set range for random shear
                            shear_range=0.,
                            # set range for random zoom
                            zoom_range=0.,
                            # set range for random channel shifts
                            channel_shift_range=0.,
                            # set mode for filling points outside the input boundaries
                            fill_mode='nearest',
                            # value used for fill_mode = "constant"
                            cval=0.,
                            # randomly flip images
                            horizontal_flip=True,
                            # randomly flip images
                            vertical_flip=False,
                            # set rescaling factor (applied before any other transformation)
                            rescale=None,
                            # set function that will be applied on each input
                            preprocessing_function=None,
                            # image data format, either "channels_first" or "channels_last"
                            data_format=None,
                            # fraction of images reserved for validation (strictly between 0 and 1)
                            validation_split=0.0)

model = ResNet_v2(input_shape=input_shape, depth=depth)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['acc'])
model.summary()
datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    epochs=epochs, callbacks=callbacks, verbose=1)