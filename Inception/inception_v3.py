### inception v3 ###
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, losses, initializers, layers, models
import numpy as np
import os


inception = InceptionV3(input_shape=(299,299,3), include_top=False)
inception.summary()

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
                                                    target_size=(299, 299),
                                                    batch_size=128,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(299, 299),
                                                        batch_size=50,
                                                        class_mode='categorical')

for layer in inception.layers[:]:
    layer.trainable = False

model = models.Sequential()
model.add(inception)
model.add(layers.AvgPool2D(pool_size=[8, 8]))
model.add(layers.Dropout(0.5, seed=77))
model.add(layers.Dense(196, activation='softmax'))

model.summary()

model.compile(optimizer=optimizers.rmsprop(lr=0.045, decay=0.9),
              loss=losses.categorical_crossentropy, metrics=['acc'])
history = model.fit_generator(train_generator, steps_per_epoch=60, epochs=70,
                              validation_data=validation_generator, validation_steps=160)