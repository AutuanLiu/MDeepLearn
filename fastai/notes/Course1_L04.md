```
Email: autuanliu@163.com
Date: 2018/05/08
```
# Notes

## 学习资料
1. [fastai_deeplearn_part1/lesson_4_x.md at master · reshamas/fastai_deeplearn_part1](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/courses/dl1/lesson_4_x.md)
2. [An Introduction to Deep Learning for Tabular Data · fast.ai](http://www.fast.ai/2018/04/29/categorical-embeddings/)
3. [Yet Another ResNet Tutorial (or not) – Apil Tamang – Medium](https://medium.com/@apiltamang/yet-another-resnet-tutorial-or-not-f6dd9515fcd7)
4. [Auto-Regressive Generative Models (PixelRNN, PixelCNN++)](https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173)
5. [Stochastic Weight Averaging — a New Way to Get State of the Art Results in Deep Learning](https://towardsdatascience.com/stochastic-weight-averaging-a-new-way-to-get-state-of-the-art-results-in-deep-learning-c639ccf36a)
6. [COCOB: An optimizer without a learning rate // teleported.in](http://teleported.in/posts/cocob/)
7. [The Cyclical Learning Rate technique // teleported.in](http://teleported.in/posts/cyclic-learning-rate/)
8. [Decoding the ResNet architecture // teleported.in](http://teleported.in/posts/decoding-resnet-architecture/)
9. [machine learning - Validation Error less than training error? - Cross Validated](https://stats.stackexchange.com/questions/187335/validation-error-less-than-training-error/187404#187404)
10. [Kaggle Planet Competition: How to land in top 4% – Towards Data Science](https://towardsdatascience.com/kaggle-planet-competition-how-to-land-in-top-4-a679ff0013ba)
11. [The Cyclical Learning Rate technique // teleported.in](http://teleported.in/posts/cyclic-learning-rate/)
12. [Network In Network architecture: The beginning of Inception // teleported.in](http://teleported.in/posts/network-in-network/)
13. [Decoding the ResNet architecture // teleported.in](http://teleported.in/posts/decoding-resnet-architecture/)
14. [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-1/#arch)


## 笔记
1. Note that CLR is very similar to Stochastic Gradient Descent with Warm Restarts (SGDR), which says, “CLR is closely-related to our approach in its spirit and formulation but does not focus on restarts.” The fastai library uses SGDR as the annealing schedule (with the idea of an LR finder from CLR).
2. As far as intuition goes, conventional wisdom says we have to keep decreasing the LR as training progresses so that we converge with time.However, counterintuitively it might be useful to periodically vary the LR between a lower and higher threshold. The reasoning is that the periodic higher learning rates within the training help the model come out of any local minimas or saddle points if it ever enters into one.
3. The takeaway is that you should not be using smaller networks because you are afraid of overfitting. Instead, you should use as big of a neural network as your computational budget allows, and use other regularization techniques to control overfitting。 不能因为害怕过拟合而使用较小的网络而应当使用较大的网路然后使用正则化技术来避免过拟合。
4. A starking trend has been to make the layers deeper, with VGG taking it to 19, and GoogLeNet taking it to 22 layers in 2014. (Note that making layers wider by adding more nodes is not preferred since it has been seen to overfit.)
5. 网络层数变化  
    
    ![网络层数变化][1]
6. It is well known that increasing the depth leads to exploding or vanishing gradients problem if weights are not properly initialized. However, that can be countered by techniques like batch normalization.
7. A Shallow network (left) and a deeper network (right) constructed by taking the layers of the shallow network and adding identity layers between them
    
    ![shallow and deeper network][3]
8. A residual is the error in a result. In essence, residual is what you should have added to your prediction to match the actual.
    
    ![residual][2]

In the diagram, x is our prediction and we want it to be equal to the Actual. However, if is it off by a margin, our residual function residual() will kick in and produce the residual of the operation so as to correct our prediction to match the actual. If x == Actual, residual(x) will be 0. The Identity function just copies x

9. ResNet arch with 34 layers.

    ![ResNet arch with 34 layers][4]


[1]: http://ozesj315m.bkt.clouddn.com/img/12-revolution-of-depth.png
[2]: http://ozesj315m.bkt.clouddn.com/img/12-residual3.png
[3]: http://ozesj315m.bkt.clouddn.com/img/12-shallow-deep.png
[4]: http://ozesj315m.bkt.clouddn.com/img/12-resnet-vgg.png
