#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：PrototypeSelectionUnder
   Description :  原型选择下采样
   Email : autuanliu@163.com
   Date：2017/12/27
"""

from collections import Counter, namedtuple

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 造一个数据集, 5 维 3 类, 无冗余, 重复特征
X, y = make_classification(n_samples=5000, n_features=5, flip_y=0.01, class_sep=1,
                           n_classes=3, n_clusters_per_class=1, random_state=0,
                           weights=[0.65, 0.3, 0.05], n_repeated=0, n_redundant=0)

# 使用LDA先做一个降维
lda = LinearDiscriminantAnalysis(n_components=2)
X = lda.fit(X, y).transform(X)

# 重采样前
print('采样前: {}'.format(Counter(y).items()))

# 可视化
sns.set(style='whitegrid')
colors = sns.color_palette('Set2', 3)

plt.figure(1)
for label, col in zip([0, 1, 2], colors):
    plt.scatter(X[y == label, 0], X[y == label, 1], color=col, label='class' + str(label))
plt.title('origin data')
plt.legend(loc='best')

# 重采样
X_new, y_new = RandomUnderSampler(random_state=1).fit_sample(X, y)

# NearMiss 的采样方式
# NearMiss-1， NearMiss-2 都很容易受到噪声的影响
X_new1, y_new1 = NearMiss(random_state=0, version=1).fit_sample(X, y)
# X_new, y_new = NearMiss(random_state=0, version=2).fit_sample(X, y)
# X_new, y_new = NearMiss(random_state=0, version=3).fit_sample(X, y)

# 采样后
print('采样前: {}'.format(Counter(y_new).items()))

plt.figure(2)
for label, col in zip([0, 1, 2], colors):
    plt.scatter(X_new[y_new == label, 0], X_new[y_new == label, 1],
                color=col, label='class' + str(label))
plt.title('new data')
plt.legend(loc='best')

# 创建模型实例
models = [('SVM', SVC(kernel='rbf')),
          ('logr', LogisticRegression(max_iter=1000)),
          ('AdaBoost', AdaBoostClassifier(random_state=0)),
          ('GDBT', GradientBoostingClassifier(random_state=0)),
          ('RF', RandomForestClassifier(random_state=0)),
          ('xgboost', xgb.XGBClassifier(random_state=0))
          ]

# 创建命名元组
model_type = namedtuple('model', ['model_name', 'model_instance'])

# 模型拟合与预测
for i in range(len(models)):
    model = model_type(*models[i])
    model.model_instance.fit(X_new, y_new)
    y_pred = model.model_instance.predict(X)
    print(model.model_name + ':\n' + 'acc: {}'.format(accuracy_score(y, y_pred)))

plt.show()
