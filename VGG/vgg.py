### VGG: building ### 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from keras import layers, models, optimizers, initializers, activations, losses
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

tf.reset_default_graph()
tf.random.set_random_seed(777)

origin_dir = "d:/data/dogs-vs-cats/train"
base_dir = "d:/data/dnc"
img_path = "d:/data/dnc/test/cats/cat.1700.jpg"

### VGG: keras building ###
model = models.Sequential()
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                        input_shape=(224, 224, 3),
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=7))) # conv1
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv2
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2)) # pool1

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv3
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv4
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2)) # pool2

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv5
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv6
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2)) # pool3

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv7
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv8
model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv9
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2)) # pool3

model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv10
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv11
model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv12
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2)) # pool4

model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv13
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv14
model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same',
                        activation=activations.relu,
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # conv15
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2)) # pool5

model.add(layers.Flatten())
model.add(layers.Dense(units=4096, activation=activations.relu,
                       kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # FC1
model.add(layers.Dense(units=4096, activation=activations.relu,
                       kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # FC2
model.add(layers.Dense(units=1000, activation=activations.relu, 
                       kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # FC3
model.add(layers.Dense(units=1, activation=activations.sigmoid,
                       kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=77))) # sigmoid
model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True),
              loss=losses.binary_crossentropy, metrics=['acc'])

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255) # pixel (0, 255) --> (0, 1)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(224, 224),
                                                        batch_size=20,
                                                        class_mode='binary')

# 1 epoch 학습 후 w 변화값 & 시각화 #
history = model.fit_generator(train_generator, steps_per_epoch=2, epochs=1,
                              validation_data=validation_generator, validation_steps=50)

model.get_weights()[0][:,:,:,0]

img = image.load_img(img_path, target_size=(224,224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0) # cnn network 3D tensor
img_tensor /= 255. # normalization

layer_outputs = [layer.output for layer in model.layers[:8]] # 하위 8개 layer 출력
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1] # w filter
    
    size = layer_activation.shape[1] # size
    n_cols = n_features // images_per_row # 팔렛트
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col*images_per_row+row] # feature map 1개
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size: (row+1) * size] = channel_image # palette
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()


### tf building ###

tf.reset_default_graph()

X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

conv1 = tf.nn.conv2d()









sess = tf.Session()