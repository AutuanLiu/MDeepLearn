# ML-Examples on Docker

Docker tag | status
--- | ---
CPU | [![Build Status][1]][2]
GPU | [![Build Status][1]][2]

这个库使用具体的例子来学习 [机器学习 + 深度学习]

使用尽可能详细的注释来写代码, 另外，相关的书面说明以及个人见解与总结会发布在 [website][3]

## Usage

1. 可以通过构建 Docker 容器的方式来进行学习(免去了环境配置的麻烦， 同时支持跨平台)
    * 通过提供的 Dockerfile 进行构建，命令为
        ```bash
        docker build -t ml-example:latest -f Dockerfile .
        ```
2. 直接 pull image
    * CPU
        ```bash
           docker pull machine-learning-on-docker:cpu
         ```
    * GPU
        ```bash
           docker pull machine-learning-on-docker:gpu
        ```
    * 运行方式
        * 参见 [docker run][4]
3. git clone(需要自己配置环境)
    ```bash
    git clone https://github.com/AutuanLiu/Machine-Learning-on-docker.git
    cd Machine-Learning-on-docker
    ```
4. [download zip file][5]  
 
## 主要目录

[持续更新中...]

* 线性模型
    * 最小二乘回归
    * Lasso 回归
    * Ridge 回归
    * 多项式回归
    * 核岭回归
    * 弹性网
    * 逻辑回归
    * LDA
    * QDA
* 分类
    * SVM
    * RandomForest
    * Gradient Boost tree
    * XGBoost
    * AdaBoost
* imbalance learn
    * over-sampling
    * under-sampling
    * Combining over-sampling and under-sampling
    * Create ensemble balanced sets
* 聚类
    * K-means
    * GMM
    * Affinity Propagation
    * SpectralClustering
    * DBSCAN
    * MeanShift
* 降维
    * PCA
    * LDA
    * SVD
* 神经网络
    * CNN
    * RNN
    * LSTM
    * DNN
    * autoEncoder
    * .

------
以上内容(不完整)都会采用 TensorFlow, Pytorch, sklearn 进行实现(进度可能有点慢, 因为我也是在学习的阶段)

`TensorFlow` 和 `Pytorch` 主要是构建深度学习框架，神经网络的，这里使用它们建立线性回归等只是说明它们可以实现这样的模型，对于机器学习
模型应该尽可能采用 `sklearn` 进行构造，毕竟**术业有专攻**

如果你觉得这个 repository 有用，并且希望丰富这个repository的内容，欢迎 PR

keras 相关代码完全使用 Tensorflow 作为后端, 因为 tensorflow 已经封装了 keras 所以，可以完全不用 安装 keras 而使用

* Pytorch 和 keras 都是 面向对象 的神经网络编程
* keras 相对于 tensorlayer 封装的更为抽象, tensorlayer 在封装时仍然可以和底层的 tensorflow 代码进行交互，相对比较透明
* 纯 tensorflow 相对还是有难度的, 但是可以结合 Tensorlayer 等进行学习, 可能会比较容易(网络结构比较清晰)
* 不过要实现自己的网络结构的话, pytorch 可能是最合适的

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
12. [skorch](https://github.com/dnouri/skorch)
13. [Deep-Learning-Boot-Camp](https://github.com/QuantScientist/Deep-Learning-Boot-Camp)
14. [pytorch welcome_tutorials](https://github.com/mila-udem/welcome_tutorials)
15. [PyTorch-Mini-Tutorials](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials)
16. [practical-pytorch](https://github.com/spro/practical-pytorch)
17. [the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch)
18. [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)
19. [imbalanced-learn documentation](http://contrib.scikit-learn.org/imbalanced-learn/stable/install.html)
20. [How to handle Imbalanced Classification Problems in machine learning?](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/)
21. [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)
22. [deep-visualization-toolbox: DeepVis Toolbox](https://github.com/yosinski/deep-visualization-toolbox)
23. [fast.ai · Making neural nets uncool again](http://www.fast.ai/)

P.S. 这也是逼迫自己写代码，实战的一种手段吧！

[1]:https://travis-ci.org/AutuanLiu/Machine-Learning-on-docker.svg?branch=master
[2]:https://travis-ci.org/AutuanLiu/Machine-Learning-on-docker
[3]:https://autuanliu.github.io/
[4]:https://github.com/AutuanLiu/ML-Docker-Env
[5]:https://github.com/AutuanLiu/Machine-Learning-on-docker/archive/master.zip
