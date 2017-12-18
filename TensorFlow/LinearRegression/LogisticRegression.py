#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：LogisticRegression
   Description :  实现逻辑回归(分类算法)
   逻辑回归可以将线性回归转换为一个二值(多值)分类器， 通过 sigmoid函数将线性回归的输出缩放到0~1之间，
   如果目标值在阈值之上则预测为1类, 否则预测为0类
   数据 https://archive.ics.uci.edu/ml/datasets/Iris
   Email : autuanliu@163.com
   Date：2017/12/9
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

from utils.normalizedFunc import min_max_normalized

# 获取数据
# 返回结果是numpy的ndarray类型的数据
# X的维度(150, 4), y的维度(150,)
X, y = load_iris(return_X_y=True)

# 这里要做一个2值分类，所以保留前100行数据
# 其是线性可分的
X = X[:100]
y = y[:100]

# 分割数据集(0.8:0.2)
# 为了结果的复现，设置种子(np, tf)
seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

# 这里不允许放回取样，replace=False
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
# 求差集
# 这里使用list或者ndarray都是可以的，但是由于数据本身是ndarray的
# 为了保持一致性，这里使用ndarray类型做索引
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]

# 归一化处理，一定放在数据集分割之后，否则测试集就会受到训练集的影响
train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)

# 开始构建模型框架
# 声明需要学习的变量及其初始化
# 这里一共有4个特征，A 的维度为(4, 1)
A = tf.Variable(tf.random_normal(shape=[4, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 定义占位符等
#  一共 4 个特征
data = tf.placeholder(dtype=tf.float32, shape=[None, 4])
target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# 声明需要学习的模型
mod = tf.matmul(data, A) + b

# 声明损失函数
# 使用sigmoid 交叉熵损失函数，先对mod结果做一个sigmoid处理，然后使用交叉熵损失函数
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target))

# 定义学习率等
learning_rate = 1e-2
batch_size = 30
iter_num = 1000

# 定义优化器
opt = tf.train.GradientDescentOptimizer(learning_rate)

# 定义优化目标
goal = opt.minimize(loss)

# 定义精确度
# 默认阈值为 0.5，直接进行四舍五入
prediction = tf.round(tf.sigmoid(mod))
# 将bool型转为float32类型，这里是矩阵操作
correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
# 求平均
accuracy = tf.reduce_mean(correct)
# 结束定义模型框架

# 开始训练模型
# 定义存储结果的变量
loss_trace = []
train_acc = []
test_acc = []

# 训练模型
for epoch in range(iter_num):
    # 生成随机batch索引，不可以重复
    # 每次生成的batch索引是不一样的
    # 原则上应当是每个epoch都对所有样本训练一次
    batch_index = np.random.choice(len(train_X), size=batch_size, replace=False)

    # 用于训练的批量数据
    batch_train_X = train_X[batch_index]
    # np.matrix()使得数据变为(len(batch_index), 1)维的数据
    batch_train_y = np.matrix(train_y[batch_index]).T
    # 开始训练
    feed = {data: batch_train_X, target: batch_train_y}
    sess.run(goal, feed)
    temp_loss = sess.run(loss, feed)
    # 这里要转化为矩阵形式，要和占位符的shape对应
    temp_train_acc = sess.run(accuracy, feed_dict={data: train_X, target: np.matrix(train_y).T})
    temp_test_acc = sess.run(accuracy, feed_dict={data: test_X, target: np.matrix(test_y).T})
    # 记录数据
    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    # 输出结果
    print('epoch: {} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
                                                                          temp_train_acc, temp_test_acc))

# 结果的可视化
# 损失函数
plt.plot(loss_trace)
plt.title('Cross Entropy Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 精确度
plt.plot(train_acc, 'b-', label='train accuracy')
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')
plt.show()
