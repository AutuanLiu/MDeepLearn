#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：RidgeRegression
   Description :  实现岭回归(带有正则项)
   L2 正则项
   Email : autuanliu@163.com
   Date：2017/12/9
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

# 获取数据
iris = load_iris()
x = np.array([d[3] for d in iris.data])
y = np.array([d[0] for d in iris.data])

# 开始定义模型的框架(这里不要出现真实的数据)
# 定义需要学习的模型参数 A, b, 并进行随机初始化操作
# 定义为 矩阵的形式 是为了统一数据的整体格式
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 建立 session并进行全局变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 定义数据的占位符
# 这里的数据的行数是可以扩充的
data = tf.placeholder(dtype=tf.float32, shape=[None, 1])
target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# 定义需要学习的模型的形式
mod = tf.matmul(data, A) + b

# 定义损失函数
# 限制斜率的系数不超过 0.9
param = tf.constant(1.)
ridge_loss = tf.reduce_mean(tf.square(A))
temp1 = tf.reduce_mean(tf.square(target - mod)) + tf.multiply(param, ridge_loss)
# 插入一个维度
loss = tf.expand_dims(temp1, axis=0)

# 定义超参数，学习率等
# 这里参数的设置很重要，直接影响最终的结果
learning_rate = 0.001
iter_num = 1500
batch_size = 50

# 定义优化器
opt = tf.train.GradientDescentOptimizer(learning_rate)

# 定义训练目标
goal = opt.minimize(loss)
# 模型的框架定义结束

# 训练模型
for step in range(iter_num):
    # 选取进行训练的 批量数据索引,注意占位符data是一个矩阵，这里要对应(None, 1)
    batch_index = np.random.choice(len(x), size=batch_size)
    # 选出的数据, 这里要是矩阵形式，保证可以正确 feed 占位符
    train_x = np.matrix(x[batch_index]).T
    train_y = np.matrix(y[batch_index]).T

    # 使用选出的数据子集进行训练
    sess.run(goal, feed_dict={data: train_x, target: train_y})

    # 输出结果
    print('epoch: {0}, A = {1}, b = {2}'.format(step + 1, sess.run(A), sess.run(b)))

# 提取结果, A 定义的维度是(1, 1)
[[slope]], [[intercept]] = sess.run(A), sess.run(b)

# 计算拟合值
fit_value = []
for i in x:
    fit_value.append(slope * i + intercept)

# 结果可视化
plt.plot(x, y, 'o', label='origin value')
plt.plot(x, fit_value, 'r-', label='predict value')
plt.xlabel('width')
plt.ylabel('length')
plt.legend(loc='best')
plt.show()
