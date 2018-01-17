#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：syntheticOver
   Description :  向上采样 SMOTE, ADASYN
   Email : autuanliu@163.com
   Date：2017/12/27
"""

import xgboost as xgb
from collections import Counter
from imblearn.over_sampling import ADASYN
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# 造一个分类数据, 附加冗余维度, 无噪声
# 2 个冗余特征， 2 个重复特征
X, y = make_classification(
    n_samples=5000,
    n_features=50,
    flip_y=0,
    class_sep=1,
    n_classes=5,
    n_clusters_per_class=1,
    random_state=0,
    weights=[0.05, 0.3, 0.03, 0.4, 0.22],
    n_informative=5,
    n_redundant=2,
    n_repeated=1)

# 做一个降维操作
pca = PCA(n_components=20)
X = pca.fit_transform(X)

# 重采样之前
print(sorted(Counter(y).items()), X.shape)

# 重采样操作
# sampler = SMOTE(random_state=5, k_neighbors=5, kind='svm', m_neighbors=10)
# or
sampler = ADASYN(random_state=1, n_neighbors=8)
X_resample, y_resample = sampler.fit_sample(X, y)
# or
# X_resample, y_resample = sampler1.fit_sample(X, y)

# 重采样之前
print(sorted(Counter(y_resample).items()))

# 拟合与预测评估
model = xgb.XGBClassifier()
model.fit(X_resample, y_resample)
y_pred = model.predict(X)

print('accuracy score: {}'.format(accuracy_score(y, y_pred)))
