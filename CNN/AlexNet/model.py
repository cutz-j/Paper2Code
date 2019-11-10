####################### ALEXNET Implementation #############################
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf #
# LRN --> BN
# Distributed Group Conv --> Conv
# input (224, 224, 3)

import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Input
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

IMAGENET_DIR = ''

##########################
### MODEL ARCHITECTURE ###
##########################
input_shape = (224, 224, 3)
img_input = Input(shape=input_shape, name='input')

# Conv 1
x = Conv2D(96, kernel_size=(11, 11), strides=4, padding='same', use_bias=False, name='conv1')(img_input)
x = BatchNormalization(name='bn1')(x)
x = Activation('relu', name='relu1')(x)
# Pool 1
x = MaxPooling2D(pool_size=(3, 3), strides=2, name='pool1')(x)

# Conv 2
x = Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', use_bias=False, name='conv2')(x)
x = BatchNormalization(name='bn2')(x)
x = Activation('relu', name='relu2')(x)
# pool 2
x = MaxPooling2D(pool_size=(3, 3), strides=2, name='pool2')(x)

# Conv 3
x = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, name='conv3')(x)
x = BatchNormalization(name='bn3')(x)
x = Activation('relu', name='relu3')(x)

# Conv 4
x = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, name='conv4')(x)
x = BatchNormalization(name='bn4')(x)
x = Activation('relu', name='relu4')(x)

# Conv 5
x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, name='conv5')(x)
x = BatchNormalization(name='bn5')(x)
x = Activation('relu', name='relu5')(x)
# pool 3
x = MaxPooling2D(pool_size=(3, 3), strides=2, name='pool3')(x)

# FC 1
x = Flatten()(x)
x = Dense(4096, name='fc1')(x)
x = Activation('relu', name='relu6')(x)
x = Dropout(0.5, name='dropout1')(x)

# FC 2
x = Dense(4096, name='fc2')(x)
x = Activation('relu', name='relu7')(x)
x = Dropout(0.5, name='dropout2')(x)

# Softmax
x = Dense(1000, name='fc3')(x)
x = Activation('softmax', name='softmax')(x)

alexnet = Model(img_input, x)
sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9)
alexnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
alexnet.summary()

###########
## Train ##
###########
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(IMAGENET_DIR, target_size=(224, 224), batch_size=128, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(IMAGENET_DIR, target_size=(224, 224), batch_size=128, class_mode='categorical')

#alexnet.fit_generator(train_generator, steps_per_epoch=712, epochs=20)