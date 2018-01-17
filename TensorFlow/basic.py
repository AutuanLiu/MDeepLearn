#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：basic
   Description : TensorFlow 基础
   Email : autuanliu@163.com
   Date：18-1-13
"""

import numpy as np
import tensorflow as tf

# 随机构造一个 线性回归 的问题
# X: 500 sample, 2 feature
# y = X*W+b, W = [0.3, -0.2], b = 0.7
data = np.float32(np.random.rand(500, 2))
target = (np.dot(data, [0.1, 0.2]) + 0.3).reshape(500, 1)


# 使用 tensorflow 构造模型结构
class model:
    """
    model: linear model class
    """

    def __init__(self, data, target, learning_rate):
        """
        构造函数
        Parameters
        ----------
        data: array like, matrix
            真实的 X
        target: array like, matrix
            真实的 y
        learning_rate: float
            学习率
        """
        X = tf.placeholder(tf.float32, shape=[None, 2], name='x_data')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='y_data')
        self.W = tf.Variable(initial_value=tf.random_normal([1, 2]))
        self.b = tf.Variable(initial_value=tf.zeros([1, 1]))
        y_pred = tf.matmul(X, tf.transpose(self.W)) + self.b
        self.loss = tf.reduce_mean(tf.square(y_pred - target))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.goal = optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.feed_dict = {X: data, y: target}

    def train(self, epoch_num):
        with tf.Session() as sess:
            # 初始化
            sess.run(self.init)
            # 训练
            for epoch in range(epoch_num):
                sess.run(self.goal, self.feed_dict)
                loss_t = sess.run(self.loss, self.feed_dict)
                W_t, b_t = sess.run((self.W, self.b))
                # or
                # W_t, b_t = sess.run((self.W, self.b))
                print(epoch, ' W_t =', W_t, ' b_t =', b_t, ' loss =', loss_t)


def main(epoch, lr=0.5):
    # 构造模型实例
    model_ins = model(data, target, lr)
    model_ins.train(epoch)


if __name__ == '__main__':
    main(3000, 0.5)
