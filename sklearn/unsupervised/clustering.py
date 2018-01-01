#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：clustering
   Description :  聚类分析
   Email : autuanliu@163.com
   Date：2018/1/1
"""

from sklearn.cluster import (
    KMeans, DBSCAN, SpectralClustering, Birch,
    AffinityPropagation, MeanShift, MiniBatchKMeans
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 数据获取与分割
X, y = make_classification(n_samples=50000, n_features=10, n_informative=10, n_redundant=0,
                           n_repeated=0, n_classes=5, n_clusters_per_class=1, flip_y=0.01,
                           class_sep=3.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# 模型
models = [
    KMeans(n_clusters=5, random_state=0, max_iter=1000),
    DBSCAN(),
    SpectralClustering(),
    Birch(),
    AffinityPropagation(),
    MeanShift(),
    MiniBatchKMeans()
]

# 拟合
for model in models:
    model.fit(X_train)
    # 预测
    # y_pred = model.predict(X_test)
    # print('error: {}/{}'.format(np.sum(y_test != y_pred), y_test.shape[0]))
