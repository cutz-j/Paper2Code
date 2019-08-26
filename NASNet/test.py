import os
os.chdir("d:/github/Paper2Code")
from NASNet.nasnet_keras import NASNet

input_shape = (32, 32, 3)
nasnet = NASNet(input_shape=input_shape)
nasnet.summary()