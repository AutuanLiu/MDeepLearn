#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：combination
   Description :  Combination of over-sampling and under-sampling
   http://contrib.scikit-learn.org/imbalanced-learn/stable/combine.html
   1. SMOTETomek
   2. SMOTEENN
   Email : autuanliu@163.com
   Date：2017/12/28
"""

from collections import Counter

import xgboost as xgb
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix

# 构造分类数据
X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1, flip_y=0.01,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=5)

# 重新采样器与模型构造
sampler = SMOTETomek(random_state=0)
sampler1 = SMOTEENN(random_state=0)
model = xgb.XGBClassifier()

# 重新采样前
print('采样前: {}'.format(sorted(Counter(y).items())))
model.fit(X, y)
y_pred0 = model.predict(X)

# 不均衡学习的情况下得到的 acc 其实是没有太大的意义的，这里只是为了做一个比较
# 重新采样后，模型的acc反而下降了，这是正常的，也是希望得到的效果
# 采样前完全可以直接将所有样本预测为最大类，这样就可以得到很大的 acc
# 混淆矩阵更清楚明了
print('采样前 acc: {}'.format(accuracy_score(y, y_pred0)))
print(confusion_matrix(y, y_pred0))

# 模型训练
for sampler_instance in [sampler, sampler1]:
    X_new, y_new = sampler_instance.fit_sample(X, y)
    print('采样后: {}'.format(sorted(Counter(y_new).items())))
    model.fit(X_new, y_new)
    y_pred = model.predict(X)
    print('acc: {}'.format(accuracy_score(y, y_pred)))
    print(confusion_matrix(y, y_pred))
