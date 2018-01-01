#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：clustering
   Description :  聚类分析
   Email : autuanliu@163.com
   Date：2018/1/1
"""

from collections import namedtuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (
    DBSCAN, SpectralClustering, Birch,
    AffinityPropagation, MeanShift, estimate_bandwidth
)
from sklearn.datasets import make_blobs
from sklearn.metrics import homogeneity_score
from sklearn.preprocessing import StandardScaler

# 聚类数据构造与标准化
X, y = make_blobs(n_samples=2000, n_features=2, cluster_std=0.5,
                  centers=3, random_state=0)
X = StandardScaler().fit_transform(X)

# 可视化
sns.set(style='whitegrid')
colors = sns.color_palette('Set2', 3)
plt.figure(1)
for color, labels in zip(colors, [0, 1, 2]):
    plt.scatter(X[y == labels, 0], X[y == labels, 1], color=color, label='class' + str(labels))
plt.title('origin data')
plt.legend(loc='best')

# 模型
bandw = estimate_bandwidth(X, quantile=0.2, n_samples=500, random_state=0)

# 谱聚类算法 用在这里是不合适的, 这里只是举例子
# assign_labels='kmeans' 可以匹配更精细的数据细节, 但是可能不稳定
# assign_labels='discretize' 策略是 100% 可以复现的, 但是它往往会产生相当规则的几何形状
models = [
    ('DDBSCAN', DBSCAN(eps=0.3, min_samples=5, metric='euclidean')),
    ('SpectralClustering', SpectralClustering(n_clusters=3, assign_labels='discretize')),
    ('Birch', Birch(n_clusters=3, threshold=0.5, compute_labels=True)),
    ('AffinityPropagation', AffinityPropagation(damping=0.6, preference=50, verbose=True,
                                                convergence_iter=50)),
    ('MeanShift', MeanShift(bandwidth=bandw, bin_seeding=True))
]

mod_tuple = namedtuple('model', ['name', 'model'])

# 拟合与评估
for i in range(len(models)):
    mod = mod_tuple(*models[i])
    # 预测
    mod.model.fit(X)
    y_pred = mod.model.labels_
    print('model/score: {}/{}\n'.format(mod.name, homogeneity_score(y, y_pred)))

plt.show()
