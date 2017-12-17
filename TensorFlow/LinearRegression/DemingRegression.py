#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：DemingRegression
   Description :  实现一个基本的Deming回归线性模型;
   使用 sklearn 的 iris 数据，x: 花瓣宽度 y: 花瓣长度， 大致满足 线性关系;
   最小二乘回归算法最小化到回归直线的竖直距离（平行于y轴），而Deming回归最小化到回归直线的垂直距离的总和，其最小化x，y两个方向的误差
   Email : autuanliu@163.com
   Date：2017/12/8
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
mod = tf.add(tf.matmul(data, A), b)

# 定义损失函数(点到直线距离的和)
fraction1 = tf.abs(target - mod)
fraction2 = tf.sqrt(tf.square(A) + 1)
# 求平均损失
loss = tf.reduce_mean(tf.truediv(fraction1, fraction2))

# 定义超参数，学习率等
# 这里参数的设置很重要，直接影响最终的结果
learning_rate = 0.05
iter_num = 200
batch_size = 50

# 定义优化器
opt = tf.train.GradientDescentOptimizer(learning_rate)

# 定义训练目标
goal = opt.minimize(loss)
# 模型的框架定义结束

# 定义用于记录结果的变量
loss_trace = []

# 训练模型
for step in range(iter_num):
    # 选取进行训练的 批量数据索引,注意占位符data是一个矩阵，这里要对应(None, 1)
    batch_index = np.random.choice(len(x), size=batch_size)
    # 选出的数据, 这里要是矩阵形式，保证可以正确 feed 占位符
    train_x = np.matrix(x[batch_index]).T
    train_y = np.matrix(y[batch_index]).T

    # 使用选出的数据子集进行训练
    sess.run(goal, feed_dict={data: train_x, target: train_y})
    temp_loss = sess.run(loss, feed_dict={data: train_x, target: train_y})

    # 输出结果
    print('epoch: {}, A = {}, b = {}, loss = {}\n'.format(step + 1, sess.run(A), sess.run(b), temp_loss))

    # 记录结果
    loss_trace.append(temp_loss)

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

# 损失函数
plt.plot(loss_trace, 'b--')
plt.title('Loss trace')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
