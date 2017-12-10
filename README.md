# ML-Examples on Docker

这个库使用具体的例子来学习 [机器学习 + 深度学习]

* 使用尽可能详细的注释来写代码
* 可以通过构建 Docker 容器的方式来进行学习(免去了环境配置的麻烦， 同时支持跨平台)
* 通过提供的 Dockerfile 进行构建，命令为
    ```shell
    docker build -t ml-example:latest -f Dockerfile .
    ```

## 主要目录
[持续更新中...]
* 线性模型
    * 最小二乘回归
    * Lasso 回归
    * Ridge 回归
    * 弹性网
    * 逻辑回归
* 分类
    * SVM
    * .
* 聚类
    * K-means
    * .
* 降维
    * PCA
    * .
* 神经网络
    * CNN
    * RNN
    * LSTM
    * autoEncoder
    * .

------
以上内容(不完整)都会采用TensorFlow, Pytorch, sklearn 进行实现，目前在做的是 TensorFlow和sklearn部分(进度可能有点慢)

TensorFlow和Pytorch主要是构建深度学习框架，神经网络的，这里使用它们建立线性回归等只是说明它们可以实现这样的模型，对于机器学习
模型应该尽可能采用 sklearn进行构造，毕竟**术业有专攻**

### 参考文献
1. [Tensorflow Machine Learning Cookbook](https://github.com/nfmcclure/tensorflow_cookbook)
2. [scikit-learn](http://sklearn.apachecn.org/cn/0.19.0/documentation.html)
3. [TensorFlow-Tutorials](https://github.com/golbin/TensorFlow-Tutorials)
4. [Simple PyTorch Tutorials Zero to ALL](https://github.com/hunkim/PyTorchZeroToAll)
5. [TensorFlow Basic Tutorial Labs](https://github.com/hunkim/DeepLearningZeroToAll)
6. [A set of examples around pytorch](https://github.com/pytorch/examples)
7. [pytorch中文网](https://ptorch.com/news/17.html)
8. [PyTorch](http://pytorch.org/)
9. [Getting Started With TensorFlow](https://www.tensorflow.org/get_started/get_started)
10. [tensorflow-zh](https://github.com/jikexueyuanwiki/tensorflow-zh)
11. [machine learning in Python](https://github.com/scikit-learn/scikit-learn)

P.S. 这也是逼迫自己写代码，实战的一种手段吧
