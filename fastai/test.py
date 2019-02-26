import fastai
from fastai import *    # Quick access to most common functionality
from fastai.vision import *    # Quick access to computer vision functionality

torch.backends.cudnn.benchmark = True
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit(1)
