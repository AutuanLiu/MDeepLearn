#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：MLP
   Description :  多层感知器
   Email : autuanliu@163.com
   Date：2017/12/30
"""

from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 不均衡数据集
X1, y1 = make_classification(
    n_samples=3000,
    n_features=10,
    n_informative=5,
    n_redundant=3,
    n_repeated=2,
    n_classes=3,
    n_clusters_per_class=1,
    flip_y=0.01,
    class_sep=0.8,
    shuffle=True,
    random_state=0,
    weights=[0.05, 0.2, 0.75])

X2, y2 = make_regression(n_samples=3000, n_features=10, n_informative=5, n_targets=1, noise=0.01, bias=3.2, random_state=0)

# 重采样
print(sorted(Counter(y1).items()))
sampler = SMOTE(random_state=0)
X1_new, y1_new = sampler.fit_sample(X1, y1)
print(sorted(Counter(y1_new).items()))

# 模型
# MLPClassifier 只支持交叉熵损失函数, 通过运行 predict_proba 方法进行概率估计
# 应用 Softmax 作为输出函数来支持多分类
model1 = MLPClassifier(
    hidden_layer_sizes=(100, 10),
    activation='tanh',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    max_iter=2000,
    early_stopping=True,
    random_state=0,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    verbose=True,
    validation_fraction=0.1,
    tol=1e-5)

# MLPRegressor 类实现了一个多层感知器(MLP), 它在使用反向传播进行训练时的输出层没有使用激活函数
# 也可以看作是使用身份函数作为激活函数. 因此, 它使用平方误差作为损失函数, 输出是一组连续值
model2 = MLPRegressor(hidden_layer_sizes=(100, 5), max_iter=2000, learning_rate_init=0.0001, learning_rate='adaptive', verbose=True, random_state=0)

# 归一化与 pipeline
scaler = StandardScaler()

model1 = make_pipeline(scaler, model1)
model2 = make_pipeline(scaler, model2)

# 拟合
model1.fit(X1_new, y1_new)
model2.fit(X2, y2)

# 预测
y1_pred = model1.predict(scaler.fit_transform(X1))
y2_pred = model2.predict(scaler.fit_transform(X2))
print(np.allclose(y1, y1_pred))
print(model1.predict_proba(X1))
print('false count: {}'.format(np.sum(y1 != y1_pred)))
print(accuracy_score(y1, y1_pred))
print(r2_score(y2, y2_pred))
