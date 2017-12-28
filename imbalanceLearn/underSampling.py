#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：underSampling
   Description : Cleaning under-sampling techniques
   TomekLinks, EditedNearestNeighbours
   CondensedNearestNeighbour
   RepeatedEditedNearestNeighbours, AllKNN
   NeighbourhoodCleaningRule, InstanceHardnessThreshold
   以上的方法 均不指定类别的样本个数，不保证每个类别的样本数是相同的
   Email : autuanliu@163.com
   Date：2017/12/28
"""

from collections import Counter

from imblearn.under_sampling import CondensedNearestNeighbour, OneSidedSelection
from imblearn.under_sampling import RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, NeighbourhoodCleaningRule
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 数据集生成
X, y = make_classification(n_samples=3000, n_features=3, flip_y=0.01, class_sep=1,
                           n_classes=3, n_clusters_per_class=1, random_state=0,
                           weights=[0.65, 0.3, 0.05], n_repeated=0, n_redundant=0)
print('采样前: {}'.format(Counter(y).items()))

# 下采样
sampler = TomekLinks(ratio='auto', random_state=0)
sampler1 = EditedNearestNeighbours(random_state=0)
sampler2 = RepeatedEditedNearestNeighbours(random_state=0, max_iter=500)
sampler3 = AllKNN(random_state=0)
sampler4 = CondensedNearestNeighbour(random_state=0)
sampler5 = OneSidedSelection(random_state=0, n_seeds_S=5)
sampler6 = NeighbourhoodCleaningRule(random_state=0)
sampler7 = InstanceHardnessThreshold(random_state=0, cv=10)

for x in [sampler, sampler1, sampler2, sampler3, sampler4, sampler5, sampler6, sampler7]:
    X_new, y_new = x.fit_sample(X, y)
    print('采样后: {}'.format(Counter(y_new).items()))
    # 拟合
    y_pred = SVC().fit(X_new, y_new).predict(X)
    print(accuracy_score(y, y_pred))

# 不重新采样的效果
y_pred1 = y_pred = SVC().fit(X, y).predict(X)
print('不重新采样的 acc: {}'.format(accuracy_score(y, y_pred)))
