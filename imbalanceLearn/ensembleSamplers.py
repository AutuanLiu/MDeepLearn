#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：ensembleSamplers
   Description :  ensemble 采样器
   EasyEnsemble, BalanceCascade, BalancedBaggingClassifier
   Email : autuanliu@163.com
   Date：2017/12/28
"""

import numpy as np
import xgboost as xgb
from collections import Counter
from imblearn.ensemble import EasyEnsemble, BalanceCascade, BalancedBaggingClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 构造分类数据
X, y = make_classification(
    n_samples=5000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=1,
    flip_y=0.01,
    weights=[0.01, 0.55, 0.44],
    class_sep=0.8,
    random_state=0)

# 重新采样器与模型构造
sampler = EasyEnsemble(random_state=0, n_subsets=10)
sampler1 = BalanceCascade(
    random_state=0,
    n_max_subset=10,
    estimator=LogisticRegression(random_state=0))
model = xgb.XGBClassifier()
sampler2 = BalancedBaggingClassifier(
    base_estimator=model, random_state=0, ratio='auto', replacement=False)

# 重新采样前
print('采样前: {}'.format(sorted(Counter(y).items())))

# 模型训练
for sampler_instance, sampler_name in zip([sampler, sampler1],
                                          ['EasyEnsemble', 'BalanceCascade']):
    X_new, y_new = sampler_instance.fit_sample(X, y)
    # 多个平衡数据集
    print('\n{} 采样后单个子集: \n{}'.format(sampler_name,
                                      sorted(Counter(y_new[0]).items())))
    acc = []
    for i in range(y_new.shape[0]):
        model.fit(X_new[i], y_new[i])
        y_pred = model.predict(X)
        acc.append(accuracy_score(y, y_pred))
    print('{} acc: {}'.format(sampler_name, np.mean(acc)))

# sampler2
sampler2.fit(X, y)
y_pred1 = sampler2.predict(X)
print('\nBalancedBaggingClassifier acc: {}'.format(accuracy_score(y, y_pred1)))
