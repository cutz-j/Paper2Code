from Deconvnet import DeConvNet

DeconvNet = DeConvNet()
DeconvNet.train(epochs=20, steps_per_epoch=500, batch_size=64)
DeconvNet.save()
#prediction = deconvNet.predict()
