#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：GaussianMixture
   Description :  高斯混合模型
   通常用于聚类分析
   Email : autuanliu@163.com
   Date：2017/12/30
"""

from sklearn.datasets import load_iris
from sklearn.metrics import homogeneity_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import train_test_split

# 数据集获取与分割
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10)

# 模型
model = GaussianMixture(n_components=3, random_state=5, max_iter=1000)
model1 = BayesianGaussianMixture(
    n_components=3,
    random_state=0,
    max_iter=1000,
    weight_concentration_prior=0.2)

# 训练
model.fit(X_train)
model1.fit(X_train)

# 预测
y_pred = model.predict(X_test)
y_pred1 = model1.predict(X_test)
print(homogeneity_score(y_test, y_pred))
print(homogeneity_score(y_test, y_pred1))
