#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：biClusteringL
   Description : 双聚类，对行列同时进行聚类
   Email : autuanliu@163.com
   Date：2018/1/1
"""

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score

data, rows, columns = make_biclusters(
    shape=(300, 300), n_clusters=5, noise=0.5, random_state=0)

model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_, (rows, columns))
print('scores: {}'.format(score))
