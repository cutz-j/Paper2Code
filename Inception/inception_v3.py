### inception v3 ###
from keras.applications.inception_v3 import InceptionV3, decode_predictions
import numpy as np


inception = InceptionV3(input_shape=(299,299,3))
inception.summary()
