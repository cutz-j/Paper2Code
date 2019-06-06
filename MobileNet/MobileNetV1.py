import keras
from keras.applications.mobilenet import MobileNet
from keras.layers import Activation, Input
from keras.models import Model

mn = MobileNet()
mn.summary()
