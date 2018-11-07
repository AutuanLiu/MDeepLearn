import fastai
from fastai import *          # Quick access to most common functionality
from fastai.vision import *   # Quick access to computer vision functionality

torch.backends.cudnn.benchmark = True
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
data.normalize(imagenet_stats)
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1, 0.01)
accuracy(*learn.get_preds())
