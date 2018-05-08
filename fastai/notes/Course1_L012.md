```
Email: autuanliu@163.com
Date: 2018/05/07
```
# Notes

## 学习资料
* dogscats resnet34: fast.ai DL lesson1.ipynb
* dogscats resnext50 architecture: lesson1-rxt50.ipynb
* Satellite Imagery (planet dataset): lesson2-image_models
* lesson2-image_models.ipynb

## broadcasting
1. [Broadcasting — NumPy v1.14 Manual](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#module-numpy.doc.broadcasting)
2. [Broadcasting semantics — PyTorch master documentation](https://pytorch.org/docs/stable/notes/broadcasting.html?highlight=broadcasting)

When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when
    
* they are equal, or
* one of them is 1

## Two tensors are “broadcastable” if the following rules hold:

* Each tensor has at least one dimension.
* When iterating over the dimension sizes, starting at the **trailing dimension**, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

## How a Picture Becomes Numerical Data?
1. an image is made up of pixels
2. each pixel is represented by a number from 0 to 255
    * White = 255
    * Black = 0 (small numbers, close to )
3. we’re working with this matrix of numbers

## What if we took convolutions and stacked them up, on top of each other?

1. We would have convolutions of convolutions, taking output of a convolution and input-ing it into another convolution?
2. That would actually not be interesting, because we're doing a linear function to another linear function.
3. What is interesting, is if we put a **non-linear** function in between.

## Neural Network is:

1. linear function followed by some sort of non-linearity
    * we can repeat that a few times
    * A common type of non-linearity: ReLU (Rectified Linear Unit) max(0, x)

2. multiply numbers (kernel by a fixed frame)
3. add the numbers up
4. put thru non-linearity (ReLU): set negative number to 0, leave positive number as is

## Filters
* can find edges, diagonals, corners
* vertical lines in the middle, left
* can find checkerboard patterns, edges of flowers, bits of text
* each layer finds multiplicatively more complex features
* dogs heads, unicycle wheels

## What's next?

Nothing, that's it.
Neural nets can approximate any function.
There are so many functions.
GPUs can do networks at scale; can do billions of operations a second.

## Data Cleaning
* we can look at data that are incorrectly classified
* Approach: build model, find out what data needs to be cleaned.
* we can look at "most correct dogs", "most correct cats"
* can look at "most incorrect"
* can look at "most uncertain predictions", sorted by how close to 0.5 the probability is

## TTA (Test Time Augmentation)
TTA simply makes predictions not just on the images in your validation set, but also makes predictions on a number of randomly augmented versions of them too (by default, it uses the original image along with 4 randomly augmented versions). It then takes the average prediction from these images, and uses that. To use TTA on the validation set, we can use the learner's TTA() method.

## Review: easy steps to train a world-class image classifier
* Enable data augmentation, and precompute=True
* Use lr_find() to find highest learning rate where loss is still clearly improving
* Train last layer from precomputed activations for 1-2 epochs
* Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
* Unfreeze all layers
* Set earlier layers to 3x-10x lower learning rate than next higher layer
* Use lr_find() again
* Train full network with cycle_mult=2 until over-fitting

And more... 
* Use lr_find() to find highest learning rate where loss is still clearly improving
* Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
* Unfreeze all layers
* Set earlier layers to 3x-10x lower learning rate than next higher layer
* Train full network with cycle_mult=2 until over-fitting

## Tips
* can set sz=64, use small size photo in beginning to get model running, and then increase the size
* most ImageNet models are trained on 224x224 or 299x299 sized images. Images in that range will work well with these algorithms.
* starting training on a few epochs with small size sz=224 and then pass in a larger size of images and continue training. This is another way to get state-of-the-art results. Increase size to 299. If I overfit with 224 size, then I'm not overfitting with 299. This method is an effective way to avoid overfitting.

## precompute=True
* started with a pre-trained network; found activations with rich features; then we add a couple of layers at the end, which start off random
* with freeze (frozen by default) and precompute=True, all we are learning is the couple of layers we've added
* with precompute=True, we actually precalculate how much does this image have the features such as eyeballs, face, etc.
* data augmentation doesn't do anything with precompute=True because we're actually showing the same exact activations every time.
* we can then set precompute=False, which means it is still only training the last couple of layers, but data augmentation is now working because it is going through and re-calculating all the activations from scratch
* finally, when we unfreeze, we can go back and change the earlier convolutional filters
* having precompute=True initially makes it faster, 10x faster. It doesn't impact the accuracy. It's just a shortcut.
* if you're showing the algorithm less images each time, then it is calculating the gradient with less images, and is less accurate
* if making batch size smaller, making algorithm more volatile; impacts the optimal learning rate.
* if you're changing the batch size by much, can reduce the learning rate by a bit.

## Architecture Types
1. resnet34 - great starting point, and often a good finishing point, doesn't have too many parameters, works well with small datasets
2. resnext - 2nd place winner in last year's ImageNet competition.
* can put a number at end to put how big it is
* resnext50 - next step after resnet34
* takes twice as long to run as resnet34
* takes 2-3x the memory as resnet34


## 相关资料
1. [DeepLearning-Lec1Notes - Part 1 - Deep Learning Course Forums](http://forums.fast.ai/t/deeplearning-lec1notes/7089)
2. [Another treat! Early access to Intro To Machine Learning videos - Part 1 - Deep Learning Course Forums](http://forums.fast.ai/t/another-treat-early-access-to-intro-to-machine-learning-videos/6826?source_topic_id=9398)
3. [Image Kernels explained visually](http://setosa.io/ev/image-kernels/)
4. [side-on 和 top-down 的区别](http://forums.fast.ai/t/wiki-lesson-2/9399/21)
5. [CNNs from different viewpoints – ImpactAI – Medium](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)
6. [Linear algebra cheat sheet for deep learning – Towards Data Science](https://towardsdatascience.com/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c)
7. [resnet PyTorch 官方实现](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
8. [A friendly introduction to Convolutional Neural Networks and Image Recognition - YouTube](https://www.youtube.com/watch?v=2-Ol7ZB0MmU)
9. [fastai_deeplearn_part1/lesson_1b_cnn_tools.md at master · reshamas/fastai_deeplearn_part1](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/courses/dl1/lesson_1b_cnn_tools.md)
10. [Universal approximation theorem - Wikipedia通用近似理论](https://en.wikipedia.org/wiki/Universal_approximation_theorem)