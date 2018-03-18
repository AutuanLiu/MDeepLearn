#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：kmeansL
   Description :  K means 聚类分析
   Email : autuanliu@163.com
   Date：2018/1/1
"""

from collections import namedtuple

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import (homogeneity_score, completeness_score,
                             v_measure_score, adjusted_rand_score,
                             adjusted_mutual_info_score)
from sklearn.preprocessing import scale

# 数据
data, target = load_digits(return_X_y=True, n_class=10)

# 标准化
data = scale(data)

# score
scores = [('homogeneity_score', homogeneity_score), ('completeness_score',
                                                     completeness_score),
          ('v_measure_score', v_measure_score), ('adjusted_mutual_info_score',
                                                 adjusted_mutual_info_score),
          ('adjusted_rand_score', adjusted_rand_score)]

met = namedtuple('metric', ['name', 'func'])

# model
pca = PCA(n_components=10, random_state=0)
pca_trans = pca.fit(data)
models = [
    KMeans(n_clusters=10, random_state=0),
    KMeans(n_clusters=10, init='random', random_state=0),
    KMeans(n_clusters=10, n_init=1, init=pca.components_, random_state=0),
    MiniBatchKMeans(n_clusters=10, random_state=0),
    MiniBatchKMeans(n_clusters=10, random_state=0, batch_size=100)
]

# fit
for model in models:
    model.fit(data)
    print()
    for index, score in enumerate(scores):
        mets = met(*scores[index])
        print(mets.name + ': {}'.format(mets.func(target, model.labels_)))
