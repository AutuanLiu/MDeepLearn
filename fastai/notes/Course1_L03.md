```
Email: autuanliu@163.com
Date: 2018/05/07
```
[Practical Deep Learning For Coders, Part 1, Lesson03.](http://course.fast.ai/index.html)

[video](https://www.youtube.com/watch?v=9C06ZPF8Uuc&feature=player_embedded)

# Lesson03 Notes
## 学习资料
1. lesson2-image_models.ipynb
2. lesson1-rxt50
3. keras_lesson1.ipynb
4. [teleported.in](http://teleported.in/)
5. [Apil Tamang – Medium](https://medium.com/@apiltamang)
6. lesson3-rossman.ipynb
7. lesson3-ross.ipynb notebook

## 符号链接 Symbolic Links
```bash
ln -s /Users/name/Downloads /Users/name/Desktop
```
How to Create and Use Symbolic Links (aka Symlinks) on a Mac

In Linux, can do ls -l dir_name  to see what the symlinks point to

## Notes
1. 设置数据的根路径
2. 卷积做的是元素相乘而不是矩阵相乘
3. PyTorch的DataLoader每次返回的是一个 minibatch 的数据
4. 数据扩充非常重要
5. if you ask for data augmentation and have precompute=True, it doesn't actually do any data augmentation, because it is using the cached non-augmented activations
6. VGG architecture, has up to 19 layers. first successful deep architecture. VGG contains a fully connected layer. Has 4,096 activations connected to a hidden layer with 4,096. 300 million weights of which 250 million are within fully connected layers
7. resnet and resnext do not have a lot of fully connected layers behind the scenes

## Structured Data
* Unstructured data: images, audio, natural language text
* Structured data: profit/loss statement, data in a spreadsheet, info about FB user, each column is structurally quite different (sex, zip code)
* structured data is what makes the world go around, though it is not presented at fancy conferences

## Freezing
* after we run an architecture (say resnet34), by default, all the layers are frozen except the last one
* when we unfreeze, we unfreeze all the layers prior to the last one
* when we "re-train" or "learn again", we are updating the weights from the pre-trained model.
* note that pre-trained weights are there from the architecture we've chosen

1. Unfreezing `learn.unfreeze()`
    * now unfreeze so we can train the whole thing
2. learn.bn_freeze Batch Norm Unfreeze
    * If you're using a bigger, deeper model like resnet50 or resnext101, on a dataset that is very similar to ImageNet; This line should be added when you unfreeze. This causes the** batch normalization moving averages** to not be updated. (more in second half of course) not supported by another library, but turns out to be important
    * if you are using an architecuture with larger than 34 suffix (resnet50, resnext101), and you're training dataset with photos similar to ImageNet (normal photos, normal size, object in middle of photo and takes up most of frame), then you should add bn_freeze. If in doubt, try with and without it.


## 相关资源
1. [Estimating an Optimal Learning Rate For a Deep Neural Network](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)
2. [Decoding the ResNet architecture](http://teleported.in/posts/decoding-resnet-architecture/)
3. [A visual and intuitive understanding of deep learning - YouTube](https://www.youtube.com/watch?time_continue=9&v=Oqm9vsf_hvU)
4. [Wiki: Lesson 3 - Part 1 - Deep Learning Course Forums](http://forums.fast.ai/t/wiki-lesson-3/9401)
5. [fastai_deeplearn_part1/lesson_3_x.md at master · reshamas/fastai_deeplearn_part1](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/courses/dl1/lesson_3_x.md)
6. [fastai_deeplearn_part1/resources.md at master · reshamas/fastai_deeplearn_part1](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/resources.md)
7. [Case Study: A world class image classifier for dogs and cats (err.., anything)](https://medium.com/@apiltamang/case-study-a-world-class-image-classifier-for-dogs-and-cats-err-anything-9cf39ee4690e)