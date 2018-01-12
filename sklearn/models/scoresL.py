#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：scoresL
   Description :  交叉验证与评估
   Email : autuanliu@163.com
   Date：2018/1/3
   # 报错信息参考以下修改：
   1. https://github.com/scikit-learn/scikit-learn/pull/9816/files
   2. https://github.com/scikit-learn/scikit-learn/blob/effbd45198c3300018403a218f1ad85858ac82dc/sklearn/preprocessing/tests/test_label.py
   3. https://github.com/scikit-learn/scikit-learn/blob/effbd45198c3300018403a218f1ad85858ac82dc/sklearn/preprocessing/label.py
"""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (cross_val_score, train_test_split,
                                     KFold)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 导入数据
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 模型
model = XGBClassifier()

# 标准化
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
model.fit(X_train, y_train)
X_test = scaler.transform(X_test)
y_prediction = model.predict(X_test)

# 评估
score1 = cross_val_score(model, X_train, y_train, cv=5)
print('cross val score: {}'.format(score1))

# 平均值与95%置信区间
print('Train Accuracy: {0:.2f} +/- {1:.2f}'.format(score1.mean(), score1.std() * 2))
print('Test Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_prediction)))

# K-fold
# RepeatedKFold 重复 K-Fold n 次
# StratifiedKFold 和 StratifiedShuffleSplit 实现的分层抽样方法
kf = KFold(n_splits=5, random_state=0, shuffle=True)
for train_idx, test_idx in kf.split(X_train):
    print(train_idx, test_idx)
    X_train1, X_test1 = X[train_idx], X[test_idx]
    y_train1, y_test1 = y[train_idx], y[test_idx]
print(X_train1, y_train1)
