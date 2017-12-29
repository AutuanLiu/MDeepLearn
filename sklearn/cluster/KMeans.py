#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：KMeans
   Description :  KMeans 聚类分析
   Email : autuanliu@163.com
   Date：2017/12/28
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = KMeans(n_clusters=3, max_iter=500)
model.fit(X)
y_pred = model.predict(X)
print(y_pred)
print(np.allclose(y, y_pred))
