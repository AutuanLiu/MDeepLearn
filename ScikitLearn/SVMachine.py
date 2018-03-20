#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：SVMachine
   Description :  支持向量机
   Email : autuanliu@163.com
   Date：2017/12/18
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.svm import SVC

sns.set(style='whitegrid')
colors = sns.color_palette("Set2", 2)

# 数据
iris = load_iris()
X, y = iris.data[:100, :2], iris.target[:100]
names = iris.target_names[:2]

# 模型建立
model = SVC(kernel='linear')

# 拟合
model.fit(X, y)

print('n support: {}\n{}'.format(model.n_support_, model.support_vectors_))
coef = model.coef_
inter = model.intercept_
x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 1000)
print(coef, inter)
y1 = (-inter - coef[0, 0] * x1) / coef[0, 1]
plt.ylim(np.min(X[:, 1]), np.max(X[:, 1]))

# 原始数据
for color, species, name in zip(colors, [0, 1], names):
    plt.scatter(X[y == species, 0], X[y == species, 1], color=color, label=name)
plt.legend(loc='best')
plt.plot(x1, y1, 'b')
plt.show()
