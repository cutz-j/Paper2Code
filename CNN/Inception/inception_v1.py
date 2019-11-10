from keras import layers, optimizers, losses
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator

def inception_module(x, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32):
    '''
    function: inception module
    '''
    x_1 = layers.Conv2D(o_1, 1, padding='same', activation='relu')(x)
    
    x_2 = layers.Conv2D(r_3, 1, padding='same', activation='relu')(x)
    x_2 = layers.Conv2D(o_3, 3, padding='same', activation='relu')(x_2)
    
    x_3 = layers.Conv2D(r_5, 1, padding='same', activation='relu')(x)
    x_3 = layers.Conv2D(o_5, 5, padding='same', activation='relu')(x_3)
    
    x_4 = layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    x_4 = layers.Conv2D(pool, 1, padding='same', activation='relu')(x_4)
    
    return layers.concatenate([x_1, x_2, x_3, x_4])


input_shape = (224, 224, 3)

input_data = layers.Input(shape=input_shape)
x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(input_data)
x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(x)
x = layers.Conv2D(192, 3, strides=1, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = inception_module(x, o_1=64, r_3=96, o_3=128, r_5=16, o_5=32, pool=32)
x = inception_module(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = inception_module(x, o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)
x = inception_module(x, o_1=160, r_3=112, o_3=224, r_5=24, o_5=64, pool=64)
x = inception_module(x, o_1=128, r_3=128, o_3=256, r_5=24, o_5=64, pool=64)
x = inception_module(x, o_1=112, r_3=144, o_3=288, r_5=32, o_5=64, pool=64)
x = inception_module(x, o_1=256, r_3=160, o_3=320, r_5=32, o_5=128, pool=128)
x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = inception_module(x, o_1=256, r_3=160, o_3=320, r_5=32, o_5=128, pool=128)
x = inception_module(x, o_1=384, r_3=192, o_3=384, r_5=48, o_5=128, pool=128)
x = layers.AvgPool2D(pool_size=(7,7), strides=1)(x)
x = layers.Flatten()(x)
output = layers.Dense(units=196, activation='softmax')(x)
output = layers.Dropout(0.4)(output)
#net = Model(input_data, output)
net.compile(optimizer=optimizers.adam(lr=0.1),
              loss=losses.categorical_crossentropy, metrics=['acc'])
net.summary()

origin_dir = "d:/data/car/car_data"
train_dir = os.path.join(origin_dir, "train")
validation_dir = os.path.join(origin_dir, "test")

#train_datagen = ImageDataGenerator(rescale=1./255, 
#                                   rotation_range=20, width_shift_range=0.1,
#                                   shear_range=0.1, zoom_range=0.1,
#                                   horizontal_flip=True, fill_mode='nearest')
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255) # pixel (0, 255) --> (0, 1)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=64,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(224, 224),
                                                        batch_size=50,
                                                        class_mode='categorical')

history = net.fit_generator(train_generator, steps_per_epoch=128, epochs=100,
                              validation_data=validation_generator, validation_steps=160)











