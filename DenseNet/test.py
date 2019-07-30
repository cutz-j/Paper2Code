from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import to_categorical
from keras.applications.densenet import DenseNet121

BATCH_SIZE = 32
NUM_CLASSES = 10
epochs = 100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

dn = DenseNet121(input_shape=(224,224,3), include_top=False)
print(dn.summary())
