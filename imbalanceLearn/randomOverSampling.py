#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：randomOverSampling
   Description :  随机向上采样
   方式: RandomOverSampler
   Email : autuanliu@163.com
   Date：2017/12/27
"""

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# 创造数据
# class_sep: 较大的值分散了簇/类，并使分类任务更容易
# flip_y：随机交换的样本部分。较大的值会在标签中引入噪音，使分类工作更加困难
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

# 可视化分析
sns.set(style='whitegrid')
colors = sns.color_palette("Set2", 3)

plt.figure(1)
for label, col in zip([0, 1, 2], colors):
    plt.scatter(X[y == label, 0], X[y == label, 1], color=col, label=label)
plt.legend(loc='best')

# 采样前
# 无噪声的情况: 5000*0.01=50, 5000*0.05=250
print(sorted(Counter(y).items()))

# 重采样
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)

print(sorted(Counter(y_resampled).items()))

# 采样后
plt.figure(2)
for label, col in zip([0, 1, 2], colors):
    plt.scatter(X_resampled[y_resampled == label, 0], X_resampled[y_resampled == label, 1], color=col, label=label)
plt.legend(loc='best')

# 拟合预测
clf = AdaBoostClassifier()
clf.fit(X_resampled, y_resampled)
y_pred = clf.predict(X)

print('错误个数: {}'.format((y_pred != y).sum()))
print(accuracy_score(y, y_pred))

plt.show()
