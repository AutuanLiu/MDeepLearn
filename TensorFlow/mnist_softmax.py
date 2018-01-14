#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：mnist_softmax
   Description :  softmax 实现 minst 预测
   Email : autuanliu@163.com
   Date：18-1-13
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def one_hot(x, dim):
    temp = np.zeros(dim)
    for index, value in enumerate(x):
        temp[index, value] = 1
    return temp


def model_train_test(x_data, y_target, lr=0.5, epoch_num=1000):
    # model
    X = tf.placeholder(D_type, shape=[None, n_features])
    y = tf.placeholder(D_type, shape=[None, n_class])
    W = tf.Variable(initial_value=tf.zeros([n_features, n_class]))
    b = tf.Variable(initial_value=tf.zeros([n_class]))
    y1 = tf.matmul(X, W) + b
    y_pred = tf.nn.softmax(y1)
    # 这里要注意 求和的方向, 维度 axis 的设置
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), axis=1), axis=0)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y1))
    goal = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    feed_dict = {X: x_data[0], y: y_target[0]}
    feed_dict1 = {X: x_data[1], y: y_target[1]}

    # accuracy
    # 这里使用 y1, y_pred 其实都可以, softmax 只是做了一个归一化的处理而已, 并不会改变大小关系
    # 返回的是索引
    y_pred_label = tf.argmax(y_pred, axis=1)
    y_label = tf.argmax(y, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_label, y_label), D_type))

    # train
    init = tf.global_variables_initializer()
    loss_trace, acc_trace = [], []
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoch_num):
            sess.run(goal, feed_dict)
            loss_t = sess.run(loss, feed_dict)
            acc_t = sess.run(accuracy, feed_dict1)
            loss_trace.append(loss_t)
            acc_trace.append(acc_t)
            print(epoch, 'loss =', loss_t, ' acc =', acc_t)
        print('mean acc =', np.mean(acc_trace))
    return loss_trace, acc_trace


if __name__ == '__main__':
    # 数据获取
    data, target = load_digits(return_X_y=True)
    n_samples, n_features = data.shape
    n_class = 10
    target = one_hot(target, dim=[n_samples, n_class])
    D_type = tf.float32
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3,
                                                                        random_state=5)

    loss_tc, acc_tc = model_train_test([data_train, data_test], [target_train, target_test], epoch_num=1500)
    plt.plot(loss_tc, label='train loss')
    plt.plot(acc_tc, label='test accuracy')
    plt.title('train loss and test accuracy')
    plt.legend(loc='best')
    plt.show()

# numpy = "*"
# tensorflow = ">=1.4.0"
# pandas = ">=0.22.0"
# scikit-learn = "==0.19.1"
# seaborn = "==0.8"
# imbalanced-learn = "*"
# tensorlayer = "*"
# keras = "*"
# xgboost = "*"
# matplotlib = "==2.1.1"
