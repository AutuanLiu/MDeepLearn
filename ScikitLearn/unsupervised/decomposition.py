#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：decomposition
   Description :  数据降维实例
   旨在说明使用方法
   Email : autuanliu@163.com
   Date：2018/1/2
"""

from collections import namedtuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.decomposition import (
    PCA, IncrementalPCA, FactorAnalysis, FastICA, KernelPCA, SparsePCA,
    MiniBatchSparsePCA, MiniBatchDictionaryLearning, DictionaryLearning)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# data
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.5,
    flip_y=0.01,
    random_state=0)

# 初始设置
sns.set(style='whitegrid')
color_series = sns.color_palette('Set2', 3)
names = 'class1', 'class2', 'class3'


# plot func
def plot_func(title, colors=color_series, class_names=names, labels=(0, 1, 2)):
    """绘图函数

    用于给降维前后的数据进行绘图，以便于做对比

    Parameters
    ----------
    :param colors: list
        列表形式的颜色集合
    :param labels: list or tuple
        列表形式的标签集合
    :param class_names: str list or tuple
        列表形式的类别名集合
    :param title: str
        绘图的 title
    Returns
    -------
    graph
        返回图像
    """
    for color, label, class_name in zip(colors, labels, class_names):
        plt.scatter(
            X[y == label, 0], X[y == label, 1], color=color, label=class_name)
    plt.title(title)
    plt.legend(loc='best')


# 转换前的可视化, 只显示前两维度的数据
plt.figure(1)
plot_func('origin data')

# KernelPCA 是非线性降维, LDA 只能用于分类降维
# ICA 通常不用于降低维度，而是用于分离叠加信号
models_list = [('LDA', LinearDiscriminantAnalysis(n_components=2)),
               ('PCA', PCA(n_components=2,
                           random_state=0)), ('PCARand',
                                              PCA(n_components=2,
                                                  random_state=0,
                                                  svd_solver='randomized')),
               ('IncrementalPCA',
                IncrementalPCA(n_components=2, batch_size=10,
                               whiten=True)), ('FactorAnalysis',
                                               FactorAnalysis(
                                                   n_components=2,
                                                   max_iter=500)),
               ('FastICA', FastICA(n_components=2,
                                   random_state=0)), ('KernelPCA',
                                                      KernelPCA(
                                                          n_components=2,
                                                          random_state=0,
                                                          kernel='rbf')),
               ('SparsePCA',
                SparsePCA(n_components=2, random_state=0,
                          verbose=True)), ('MiniBatchSparsePCA',
                                           MiniBatchSparsePCA(
                                               n_components=2,
                                               verbose=True,
                                               batch_size=10,
                                               random_state=0)),
               ('DictionaryLearning',
                DictionaryLearning(
                    n_components=2, verbose=True,
                    random_state=0)), ('MiniBatchDictionaryLearning',
                                       MiniBatchDictionaryLearning(
                                           n_components=2,
                                           batch_size=5,
                                           random_state=0,
                                           alpha=0.1))]

model = namedtuple('models', ['mod_name', 'mod_ins'])

for i in range(len(models_list)):
    mod = model(*models_list[i])
    if mod.mod_name == 'LDA':
        mod.mod_ins.fit(X, y)
        X_new = mod.mod_ins.transform(X)
    else:
        X_new = mod.mod_ins.fit_transform(X)
    plt.figure(i + 2)
    plot_func(mod.mod_name + ' transformed data')
    print(mod.mod_name + 'finished!')

plt.show()
