#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：NN
   Description :  最近邻
   Email : autuanliu@163.com
   Date：2017/12/22
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree

# 无监督
# 找到两组数据集中最近邻点的简单任务
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
distances, indices = nbrs.kneighbors(X)
print(indices, distances)
# 生成一个稀疏图来标识相连点之间的连接情况
print(nbrs.kneighbors_graph(X).toarray())

plt.plot(X, 'o')
plt.show()

# KD tree
kdt = KDTree(X, leaf_size=30, metric='euclidean')
res = kdt.query(X, k=2, return_distance=False)
print(res)

# KNN 分类
# clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
# clf.fit(X, y)
