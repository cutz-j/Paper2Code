import os
os.chdir("d:/github/Paper/ZFNet")
from Deconvnet import DeConvNet

DeconvNet = DeConvNet()
DeconvNet.train(epochs=10, steps_per_epoch=1000, batch_size=128)
DeconvNet.save()
#prediction = deconvNet.predict()
