#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：PrototypeGenerationUnder
   Description :  原型生成下采样
   ClusterCentroids 使用 k-means 来减少样本数量， 每个类将与K-means方法的质心合成，而不是原始样本
   Email : autuanliu@163.com
   Date：2017/12/27
"""

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from collections import Counter, namedtuple
from imblearn.under_sampling import ClusterCentroids
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# 和 randomOverSampling 使用同样的原始数据
X, y = make_classification(
    n_samples=5000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=1,
    flip_y=0.01,
    weights=[0.01, 0.05, 0.94],
    class_sep=0.8,
    random_state=10)

# 下采样前
print(sorted(Counter(y).items()))

sns.set(style='whitegrid')
colors = sns.color_palette("Set2", 3)

plt.figure(1)
for label, col in zip([0, 1, 2], colors):
    plt.scatter(X[y == label, 0], X[y == label, 1], color=col, label='class' + str(label))
plt.title('origin data')
plt.legend(loc='best')

# ClusterCentroids 下采样
# 先创建实例
sampler = ClusterCentroids(random_state=0)

# 采样
X_resample, y_resample = sampler.fit_sample(X, y)

# 下采样后
print(sorted(Counter(y_resample).items()))

plt.figure(2)
for label, col in zip([0, 1, 2], colors):
    plt.scatter(
        X_resample[y_resample == label, 0], X_resample[y_resample == label, 1], color=col, label='class' + str(label))
plt.title('resample data')
plt.legend(loc='best')

# 创建model实例, 4 个模型
model_name = ['GradientBoostingClassifier', 'AdaBoostClassifier', 'XGBClassifier', 'RandomForestClassifier']

model_instance = [GradientBoostingClassifier(), AdaBoostClassifier(), xgb.XGBClassifier(), RandomForestClassifier()]
models = namedtuple('models', ['name', 'model'])

# 其他方式：使用字典传入参数 **dict

# fit
for index in range(4):
    mod = models(model_name[index], model_instance[index])
    mod.model.fit(X_resample, y_resample)

    # predict
    y_pred = mod.model.predict(X)

    # 评估
    print(mod.name + ': ')
    print('错误个数: {} 准确度: {}\n'.format((y_pred != y).sum(), accuracy_score(y, y_pred)))

plt.show()
