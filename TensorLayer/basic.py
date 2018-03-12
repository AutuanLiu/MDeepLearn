#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：basic
   Description :  tensorflow and tensorlayer basic
   Email : autuanliu@163.com
   Date：2018/3/11
"""

import tensorflow as tf

x = tf.constant([1, 2, 3, 4], shape=[2, 2])
y = tf.constant([5, 6, 7, 8], shape=[2, 2])
# Session() 必须通过 sess 运行
with tf.Session() as sess:
    print(sess.run(x))
    # 矩阵乘法
    print(sess.run(x @ y))
    print(sess.run(tf.matmul(x, y)))
    # 矩阵点乘，对应元素相乘
    print(sess.run(tf.multiply(x, y, name='multiply')))
    print(sess.run(x * y))

# 获取 tensor 的维度
print(x.get_shape())
# 获取数据类型
print(x.dtype)
print(x.get_shape().as_list())

# InteractiveSession() 可以直接运行而不通过 sess
sess1 = tf.InteractiveSession()
print(x @ y)
# 可以不通过 sess 运行, 也可以使用 sess
print((x @ y).eval())
print(x.eval())

# another example
x1 = tf.placeholder(tf.float32, shape=[None, 3], name='x')
w = tf.Variable(initial_value=[[1., 3., 4.]], dtype=tf.float32)
# 定义操作函数
y = x1 @ (tf.transpose(w))
# or
# y = tf.matmul(x1, w, transpose_b=True)
# 初始化参数
w.initializer.run()
print('\ny res:\n {}'.format(
    y.eval(feed_dict={x1: [[1., 3., 5.], [2., 4., 7.]]}, session=sess1)))
# 最后一定要关闭 session
sess1.close()
