#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：LinearDiscriminant
   Description :  线性判别用于降维和PCA降维的对比
   线性判别分析是有利于分类的，而PCA只是找到了最能代表整体数据的维度
   通过将样本向判别边界投影，可以最大程度的区分样本类别
   PCA是寻找有效表示的主轴方向而线性判别分析是寻找有效分类的方向；
   线性判别分析的依据是：类间散布和类内散布的比值达到最大，
   同一类的距离越近越好，不同类的距离越远越好
   Email : autuanliu@163.com
   Date：2017/12/18
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 导入数据, 这里要获得 target 的名字，所以不设 return_X_y
iris = load_iris()
X = iris.data
y = iris.target
target_name = iris.target_names

# PCA
# 实例化
pca = PCA(n_components=2)
X_trans1 = pca.fit_transform(X)

# LinearDiscriminantAnalysis, 有监督
lda = LinearDiscriminantAnalysis(n_components=2)
X_trans2 = lda.fit_transform(X, y)

# 预测测试, 对应的标签是 0, 1, 2, 1
test = [[4.0, 3.1, 1.1, 0.1],
        [6.7, 3.1, 4.1, 1.4],
        [7.1, 3.2, 6.1, 1.9],
        [6.3, 2.9, 4.2, 1.4],
        ]
prediction = lda.predict(test)
print('predict result: {}'.format(prediction))

# 主成分能够解释整体所占的比例，也即特征值最大的两个特征向量对应的特征
print('first two components ratio: {}'.format(pca.explained_variance_ratio_))

# 可视化
colors = ['red', 'turquoise', 'orange']
lw = 2

plt.figure(1)
for color, species, name in zip(colors, [0, 1, 2], target_name):
    plt.scatter(X[y == species, 0], X[y == species, 1], color=color, alpha=0.8, lw=lw, label=name)
plt.title('origin data')
plt.legend(loc='best')

plt.figure(2)
for color, species, name in zip(colors, [0, 1, 2], target_name):
    plt.scatter(X_trans1[y == species, 0], X_trans1[y == species, 1], color=color, alpha=0.8, lw=lw, label=name)
plt.title('PCA transformed data')
plt.legend(loc='best')

plt.figure(3)
for color, species, name in zip(colors, [0, 1, 2], target_name):
    plt.scatter(X_trans2[y == species, 0], X_trans2[y == species, 1], color=color, alpha=0.8, lw=lw, label=name)
plt.title('LDA transformed data')
plt.legend(loc='best')
plt.show()
