#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：class_local
   Description :  catboost 使用本地数据分类实例
   Email : autuanliu@163.com
   Date：2018/3/19
"""

from catboost import Pool, CatBoost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 获取数据
data, target = load_iris(return_X_y=True)

# 训练集与测试集分割
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)
train, test = Pool(X_train, y_train), Pool(X_test)

# 构建模型
config = [{
    'iterations': 800,
    'learning_rate': 0.03,
    'depth': 6,
    'loss_function': 'MultiClass',
    'logging_level': 'Verbose',
    'random_seed': 10,
    'metric_period': 5,
    'train_dir': './boost/CatBoost/logs'
}, None]

# unpacking 的形式传入
model = CatBoost(*config)

# train
model.fit(train)

# make the prediction using the resulting model
preds_class = model.predict(test, prediction_type='Class')
preds_raw = model.predict(test, prediction_type='RawFormulaVal')
preds_prob = model.predict(test, prediction_type='Probability')
print(f'Class = \n{preds_class}')
print(f'RawFormulaVal = \n{preds_raw}')
print(f'Probability = \n{preds_prob}')

# other metrics
acc = accuracy_score(y_test, preds_class)
print(f'acc is: {acc}')
