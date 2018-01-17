#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：test.py
   Description :  
   Email : autuanliu@163.com
   Date：2017/12/20
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

# 获取原始数据
iris = load_iris()
x = np.array([d[3] for d in iris.data])
y = np.array([d[0] for d in iris.data])

# 这里想要构造的模型是 y = A*x + b 的形式，所以，这里我们要学习的参数是 A, b
# 我们将它们声明为需要学习的变量, 并进行随机初始化(为后面的梯度下降做准备)
# 这里A，b 都只是一个数，所以声明为 1*1(TensorFlow的大部分操作为矩阵操作)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 全局初始化操作
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 定义占位符
# 为了程序的通用性，这里不定义占位符的具体行数
data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 定义将要学习的模型形式
# 这里要注意乘法的顺序，保证矩阵是可以运算的
# (None, 1) * (1, 1) 相反，便不可计算
mod = tf.add(tf.matmul(data, A), b)

# 定义损失函数， L2(需要求平均值)
loss = tf.reduce_mean(tf.square(target - mod))

# 定义超参数等
learning_rate = 0.05
iter_num = 200
batch_size = 25

# 定义优化器
opt = tf.train.GradientDescentOptimizer(learning_rate)
# 优化目标: 用优化器迭代出参数，目标是使得 loss 最小
goal = opt.minimize(loss)

# 定义用于记录结果的变量
loss_trace = []

# 训练模型
for step in range(iter_num):
    # 选取进行训练的 批量数据索引
    index = np.random.choice(len(x), size=batch_size)
    # 选出的数据, 这里要是矩阵形式，保证可以正确 feed 占位符
    # numpy 是行向量，TensorFlow 使用列向量
    train_x = np.matrix(x[index]).T
    train_y = np.matrix(y[index]).T

    # 使用选出的数据子集进行训练
    sess.run(goal, feed_dict={data: train_x, target: train_y})
    temp_loss = sess.run(loss, feed_dict={data: train_x, target: train_y})

    # 输出结果
    print('epoch: {}, A = {}, b = {}, loss = {}\n'.format(
        step + 1, sess.run(A), sess.run(b), temp_loss))

    # 记录结果
    loss_trace.append(temp_loss)

# 提取结果, A 定义的维度是(1, 1)
[[slope]], [[intercept]] = sess.run(A), sess.run(b)

# 计算拟合值
fit_value = []
for i in x:
    fit_value.append(slope * i + intercept)
