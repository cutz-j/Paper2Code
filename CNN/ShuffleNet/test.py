import os
os.chdir("d:/github/Paper2Code")
from ShuffleNet.shufflenet import ShuffleNet
from keras import Input

input_shape = (224, 224, 3)
sn = ShuffleNet(input_shape=input_shape, groups=4)
