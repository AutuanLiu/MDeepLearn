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
8. [fastai/cifar10.ipynb at master · fastai/fastai](https://github.com/fastai/fastai/blob/master/courses/dl1/cifar10.ipynb)
9. [Kaggle Planet Competition: How to land in top 4% – Towards Data Science](https://towardsdatascience.com/kaggle-planet-competition-how-to-land-in-top-4-a679ff0013ba)
10. [Deep Learning 2: Part 1 Lesson 2 – Hiromi Suenaga – Medium](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-2-eeae2edd2be4)
11. [Annotated notebook](http://forums.fast.ai/uploads/default/original/2X/b/b01dffa62debfb8450fb9a3969d650645c54a3aa.pdf)
12. [DeepLearning-LecNotes2 - Part 1 - Deep Learning Course Forums](http://forums.fast.ai/t/deeplearning-lecnotes2/7515/10)

## 符号链接 Symbolic Links
```bash
ln -s /Users/name/Downloads /Users/name/Desktop
```
How to Create and Use Symbolic Links (aka Symlinks) on a Mac

In Linux, can do ls -l dir_name  to see what the symlinks point to

## make the sub directories
```sh
mkdir -p data/camelhorse/{train,valid}/{camel,horse}

# split original data into train/test
shuf -n 68 -e data/camelhorse/camels/* | xargs -i cp {} data/camelhorse/train/camel
shuf -n 68 -e data/camelhorse/horses/* | xargs -i cp {} data/camelhorse/train/horse
shuf -n 33 -e data/camelhorse/camels/* | xargs -i cp {} data/camelhorse/valid/camel
shuf -n 33 -e data/camelhorse/horses/* | xargs -i cp {} data/camelhorse/valid/horse


ls ~/data/camelhorse/camels | wc -l
ls ~/data/camelhorse/horses | wc -l
ls ~/data/camelhorse/train/camel | wc -l
ls ~/data/camelhorse/train/horse | wc -l
ls ~/data/camelhorse/valid/camel | wc -l
ls ~/data/camelhorse/valid/horse | wc -l
```

## 7z 文件
```bash
sudo apt-get install p7zip-full
7z e test.json.7z
7z e train.json.7z 
7z e sample_submission.csv.7z 
```

```
ls -alt
wc -l *
```


## Notes
1. 设置数据的根路径
2. 卷积做的是元素相乘而不是矩阵相乘
3. PyTorch的DataLoader每次返回的是一个 minibatch 的数据
4. 数据扩充非常重要
5. if you ask for data augmentation and have precompute=True, it doesn't actually do any data augmentation, because it is using the cached non-augmented activations
6. VGG architecture, has up to 19 layers. first successful deep architecture. VGG contains a fully connected layer. Has 4,096 activations connected to a hidden layer with 4,096. 300 million weights of which 250 million are within fully connected layers
7. resnet and resnext do not have a lot of fully connected layers behind the scenes
8. When you train the model, the forward pass goes through all the layers. But when you calculate an error and do backpropagation, you update only weights of layers that are “unfrozen” and don’t change weights in “frozen” layers
9. http://forums.fast.ai/t/wiki-lesson-2/9399/56?u=liu
10. [a higher learning rate will push the weights around more - and therefore will give less credence to the pretrained weights.](http://forums.fast.ai/t/wiki-lesson-2/9399/62?u=liu)
11. [machine learning - Validation Error less than training error? - Cross Validated](https://stats.stackexchange.com/questions/187335/validation-error-less-than-training-error/187404#187404)
12. [Hiromi Suenaga – Medium](https://medium.com/@hiromi_suenaga)
13. [Case Study: A world class image classifier for dogs and cats (err.., anything)](https://medium.com/@apiltamang/case-study-a-world-class-image-classifier-for-dogs-and-cats-err-anything-9cf39ee4690e)
14. [Convolutional Neural Network in 5 minutes – Hacker Noon](https://hackernoon.com/convolutional-neural-network-in-5-minutes-8f867eb9ca39)
15. [Visualizing Learning rate vs Batch size](https://miguel-data-sc.github.io/2017-11-05-first/)
16. [A friendly introduction to Convolutional Neural Networks and Image Recognition - YouTube](https://www.youtube.com/watch?v=2-Ol7ZB0MmU)
17. [computer vision - Convolutional Neural Networks - Multiple Channels - Stack Overflow](https://stackoverflow.com/questions/27728531/convolutional-neural-networks-multiple-channels)
18. [Wiki: Lesson 3 - Part 1 - Deep Learning Course Forums](http://forums.fast.ai/t/wiki-lesson-3/9401)
19. As well as looking at the overall metrics, it's also a good idea to look at examples of each of:
    1.A few correct labels at random 随机查看正确预测
    2.A few incorrect labels at random 随机查看不正确的预测
    3.The most correct labels of each class (ie those with highest probability that are correct) 正确率较高的预测
    4.The most incorrect labels of each class (ie those with highest probability that are incorrect) 错误率较高的预测
    5.The most uncertain labels (ie those with probability closest to 0.5). 不确定的预测

20. 代码一定要越简单越好
21. **使用 CLR 寻找最优的学习率 使用SGDR来更新学习率**
22. 学习率设置组

**Note that the other layers have *already* been trained to recognize imagenet photos (whereas our final layers where randomly initialized), so we want to be careful of not destroying the carefully tuned weights that are already there.**

Generally speaking, the earlier layers (as we've seen) have **more general-purpose features**. Therefore we would expect them to need *less fine-tuning* for new datasets. For this reason we will *use different learning rates for different layers*: the first few layers will be at 1e-4, the middle layers at 1e-3, and our FC layers we'll leave at 1e-2 as before. We refer to this as *differential learning rates*, although there's no standard name for this techique in the literature that we're aware of. 前面的学习率是更通用的所以我们希望更少的调整所以给他们的学习率设置较小 通常认为学习率小的话对应着对他们原来权重的信任度较大

23. The learning rates of the earlier layers are fixed at the same multiples of the final layer rates as we initially requested。 我们一直调整的是最后的一层的学习率 而之前层次的学习率一直和最后层保持着一定的倍数关系。
24. 预测的时候也要做扩充 然后使用平均值作为最后的预测结果

19. 代码
```python
os.listdir(f'{PATH}valid') # 查看当前文件夹下的内容
np.where()
np.choice()
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.stack((a, b), axis=-1)
array([[1, 2],
       [2, 3],
       [3, 4]])


# 画图工具
def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
```

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
8. [Keras Model for Beginners (0.210 on LB)+EDA+R&D | Kaggle](https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d)