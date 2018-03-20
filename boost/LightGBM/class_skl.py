#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：class
   Description :  lightGBM 分类实例 with early-stopping sklearn API
   1. https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
   Email : autuanliu@163.com
   Date：2018/3/20
"""
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 获取数据
data, target = load_digits(n_class=10, return_X_y=True)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data / 255, target, test_size=0.2)

# 构建模型
# 参数设置
params = {
    'objective': 'multiclass', 
    'num_iterations': 193, 
    'num_leaves': 31,
    'learning_rate': 0.1,
}
gbm = LGBMClassifier(**params)

# 训练
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='multi_logloss', early_stopping_rounds=15)

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
print(f'Best iterations: {gbm.best_iteration_}')
print(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
