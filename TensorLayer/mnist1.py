#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：mnist1
   Description :  mnist dataset exp
   Email : autuanliu@163.com
   Date：2018/3/12
"""

import tensorflow as tf
import tensorlayer as tl

# 定义 Session
# Session 允许计算图或者图的一部分，为这次计算分配资源并且保存中间结果的值和变量
sess = tf.InteractiveSession()
# 获取数据 28*28=784
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784), 
path="../datasets/tldata")


