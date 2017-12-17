#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：ElasticNetRegression
   Description :  实现弹性网回归，其是介于lasso回归和ridge回归之间的一种算法
   同时使用L1，L2正则化，这会导致损失函数收敛变慢
   * 使用滑板长度，花瓣宽度，花萼宽度来预测花萼长度 *
   Email : autuanliu@163.com
   Date：2017/12/9
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

# 构造数据(多特征)
# 这里的 x 本身就已经构造成ndarray矩阵的形式，但是 y 仍为向量形式
x = np.array([[d[1], d[2], d[3]] for d in iris.data])
y = np.array([d[0] for d in iris.data])

# 开始构建框架
# 变量声明与初始化
# 这里是 3 个特征，所以维度是 (3, 1)
A = tf.Variable(tf.random_normal(shape=[3, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 占位符的创建
# 注意这里的维度
data = tf.placeholder(dtype=tf.float32, shape=[None, 3])
target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# 定义学习率等
learning_rate = 0.001
iter_num = 500
batch_size = 50

# 定义模型的形式
mod = tf.matmul(data, A) + b

# 定义损失函数(最小二乘 + L1正则项 + L2正则项)
# 定义正则系数
alpha1 = tf.constant(0.9)
alpha2 = tf.constant(0.8)
least_square_term = tf.reduce_mean(tf.square(target - mod))
L1_term = tf.reduce_mean(tf.abs(A))
L2_term = tf.reduce_mean(tf.square(A))
temp = least_square_term + tf.multiply(alpha1, L1_term) + tf.multiply(alpha2, L2_term)
print(type(temp))
loss = tf.expand_dims(temp, axis=0)
print(type(loss))
# 定义优化器
opt = tf.train.GradientDescentOptimizer(learning_rate)

# 定义目标
goal = opt.minimize(loss)
# 框架定义结束

# 开始训练模型
loss_trace = []
for epoch in range(iter_num):
    # 随机产生训练数据的索引
    batch_index = np.random.choice(len(x), size=batch_size)
    # 注意这里的维度，本身就是一个矩阵形式
    train_x = x[batch_index]
    train_y = np.matrix(y[batch_index]).T
    # feed
    sess.run(goal, feed_dict={data: train_x, target: train_y})
    temp_loss = sess.run(loss, feed_dict={data: train_x, target: train_y})
    loss_trace.append(temp_loss)
    # 输出结果
    print('epoch: {}   loss = {}'.format(epoch + 1, temp_loss))

# 可视化结果
plt.plot(loss_trace)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
