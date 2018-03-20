#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：class
   Description :  lightGBM 分类实例 with early-stopping
    1. LightGBM can use categorical feature directly (without one-hot coding)
    2. The LightGBM Python module is able to load data from: 
        1. libsvm/tsv/csv/txt format file
        2. Numpy 2D array, pandas object
        3. LightGBM binary file
    3. https://lightgbm.readthedocs.io/en/latest/Parameters.html
    4. https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api
    5. Log loss, aka logistic loss or cross-entropy loss
   Email : autuanliu@163.com
   Date：2018/3/20
"""
import lightgbm as lgb, numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 获取数据
data, target = load_digits(n_class=10, return_X_y=True)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data / 255, target, test_size=0.2)

# lgb 格式数据
train = lgb.Dataset(X_train, label=y_train)
#  If this is Dataset for validation, training data should be used as reference
test = lgb.Dataset(X_test, label=y_test, reference=train)

# 参数设置
params = {
    'task': 'train',
    'boosting_type': 'gbdt', 
    'objective': 'multiclass', 
    'metric': 'multi_logloss',
    'metric_freq': 5, 
    'num_class': 10, 
    'num_iterations': 200, 
    'num_leaves': 31,
    'learning_rate': 0.1,
}

# train
gbm = lgb.train(params, train, valid_sets=test, early_stopping_rounds=15)

# save model to file
gbm.save_model('./boost/LightGBM/model.txt')

# predict
print(f'Best iterations: {gbm.best_iteration}')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval
n_class = np.argmax(y_pred, axis=1)
print(y_test, n_class)
print(accuracy_score(y_test, n_class))
