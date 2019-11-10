import os
os.chdir("d:/github/Paper2Code")
from ResNEXT.resnext import ResNEXT

input_shape = (224, 224, 3)
resnext = ResNEXT(input_shape=input_shape)
resnext.summary()