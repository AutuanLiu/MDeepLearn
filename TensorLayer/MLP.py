#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：MLP
   Description :  多层感知机模型(带有 Dropout 层), 使用 fashion-MNIST 模型
   Email : autuanliu@163.com
   Date：2018/3/5
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorlayer as tl

# 获取数据
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784), path='../datasets/tldata')
# placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='sample')
y = tf.placeholder(
    tf.int64, shape=[
        None,
    ], name='label')
sess = tf.InteractiveSession()


# 模型定义
def mlp(x, is_train=True, reuse=False):
    # 所有参数名字都以 MLP 作为前缀
    with tf.variable_scope('MLP', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DropoutLayer(net, keep=0.8, is_fix=True, is_train=is_train, name='dropout_layer1')
        net = tl.layers.DenseLayer(net, n_units=800, act=tf.nn.relu, name='dense1')
        net = tl.layers.DropoutLayer(net, keep=0.5, is_fix=True, is_train=is_train, name='dropout_layer2')
        net = tl.layers.DenseLayer(net, n_units=800, act=tf.nn.relu, name='dense2')
        net = tl.layers.DropoutLayer(net, keep=0.5, is_fix=True, is_train=is_train, name='dropout_layer3')
        net = tl.layers.DenseLayer(net, n_units=10, act=tf.identity, name='output')
    return net


# 训练模型
# model define
net_train = mlp(x, is_train=True, reuse=False)
net_test = mlp(x, is_train=False, reuse=True)

# loss and acc
y_pred = net_train.outputs
cost = tl.cost.cross_entropy(y_pred, y, name='train_cost')
# test
y_test = net_test.outputs
cost_test = tl.cost.cross_entropy(y_test, y, name='test_cost')
correct = tf.equal(tf.argmax(y_test, 1), y)
acc = tf.reduce_mean(tf.cast(correct, tf.float32))
# optimizer
train_params = tl.layers.get_variables_with_name('MLP', train_only=True, printable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, var_list=train_params)

# initial
tl.layers.initialize_global_variables(sess)
n_epoch = 1000
batch_size = 64
net_train.print_layers()
train_loss = []
train_acc = []
val_loss = []
val_acc = []
for epoch in range(n_epoch):
    for X1, y1 in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        feed = {x: X1, y: y1}
        sess.run(optimizer, feed_dict=feed)
        err, ac = sess.run([cost_test, acc], feed_dict=feed)
    # 取最后一个(实际上应当取平均值)
    train_loss.append(err)
    train_acc.append(ac)
    print("train loss: {:5f}".format(err))
    print("train acc: {:5f}".format(ac))
    for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
        err, ac = sess.run([cost_test, acc], feed_dict={x: X_val_a, y: y_val_a})
    val_loss.append(err)
    val_acc.append(ac)
    print('epoch: {}'.format(epoch + 1))
    print("val loss: {:5f}".format(err))
    print("val acc: {:5f}".format(ac))

# 可视化
plt.figure(1)
plt.plot(train_loss, label='train_loss')
plt.plot(train_acc, label='train_acc')
plt.legend(loc='best')
plt.figure(2)
plt.plot(val_loss, label='val_loss')
plt.plot(val_acc, label='val_acc')
plt.legend(loc='best')
plt.show()
