#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：pipelineUsage
   Description : 使用pipeline 执行
   Email : autuanliu@163.com
   Date：2017/12/28
"""

import xgboost as xgb
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 生成数据集
X, y = make_classification(n_features=5, n_informative=2, n_redundant=0,
                           flip_y=0.01, n_classes=3, class_sep=1.25,
                           n_clusters_per_class=1, weights=[0.3, 0.05],
                           n_samples=5000, random_state=10)

# 线性判别降维
lda = LinearDiscriminantAnalysis(n_components=2)

# 重采样
enn = EditedNearestNeighbours()

# 分类器
xgb_tree = xgb.XGBClassifier()

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

# 创建 pipeline
pipeline = make_pipeline(lda, enn, xgb_tree)

# 拟合与预测
pipeline.fit(X_train, y_train)
y_hat = pipeline.predict(X_test)

print(classification_report(y_test, y_hat))
print(confusion_matrix(y_test, y_hat))
