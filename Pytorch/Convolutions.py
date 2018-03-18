#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name： Convolutions
   Description :  卷积基础
   Email : autuanliu@163.com
   Date：2018/3/17
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import correlate
from sklearn.datasets import fetch_mldata


def plot(im, interp=False):
    im = np.array(im)
    f = plt.figure(figsize=(3,6), frameon=True)
    plt.imshow(im, interpolation=None if interp else 'none')
    plt.show()


# fetch MNIST datasets
mnist = fetch_mldata('mnist-original')
data, target = mnist.data, mnist.target
data, target = data.reshape(-1, 28, 28), target.astype(int)

# 归一化处理
data = data/255
plot(data[0])
# zoom in on part of the image
plot(data[0, 0:14, 8:22])

# Edge Detection
top = [[-1,-1,-1],
     [ 1, 1, 1],
     [ 0, 0, 0]]

plot(top)

# filter
dim = np.index_exp[10:28,3:13]
corrtop = correlate(data[0], top)
plot(corrtop)