#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：NonLinearSVM
   Description :  非线性支持向量机
   http://sklearn.apachecn.org/cn/0.19.0/auto_examples/svm/plot_svm_nonlinear.html#sphx-glr-auto-examples-svm-plot-svm-nonlinear-py
   这是一个尝试，数据集暂缺
   Email : autuanliu@163.com
   Date：2017/12/21
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.svm import NuSVC

sns.set(style='whitegrid')
colors = sns.color_palette("Set2", 2)

# 数据
iris = load_iris()
X, y = iris.data[50:, :2], iris.target[50:]
names = iris.target_names[1:]

# 模型建立
model = NuSVC(kernel='rbf')

# 拟合
model.fit(X, y)

print('n support: {}\n{}'.format(model.n_support_, model.support_vectors_))

# 原始数据
for color, species, name in zip(colors, [1, 2], names):
    plt.scatter(
        X[y == species, 0], X[y == species, 1], color=color, label=name)
plt.legend(loc='best')
plt.show()
