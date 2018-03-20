#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：class
   Description :  catboost 分类实例
   1. https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/#loss-functions
   2. https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/#python-reference_parameters-list
   3. https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostclassifier-docpage/
   Email : autuanliu@163.com
   Date：2018/3/19
"""
from catboost import CatBoostClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 获取数据
data, target = load_digits(n_class=10, return_X_y=True)
data = data / 255

# 训练集与测试集分割
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)

# 构建模型
config = {
    'iterations': 800,
    'learning_rate': 0.03,
    'depth': 6,
    'loss_function': 'MultiClass',
    'logging_level': 'Verbose',
    'random_seed': 120,
    'nan_mode': 'Min',
    'calc_feature_importance': True,
    'leaf_estimation_method': 'Gradient',
    'l2_leaf_reg': 2,
    'fold_len_multiplier': 1.2,
    'od_type': 'IncToDec',
    'metric_period': 5,
    'train_dir': './boost/CatBoost/logs'
}

# unpacking 的形式传入参数
model = CatBoostClassifier(**config)

# train
model.fit(X_train, y_train)

# make the prediction using the resulting model
preds_class = model.predict(X_test, prediction_type='Class')
preds_proba = model.predict_proba(X_test)
# or
# preds_proba = model.predict(X_test, prediction_type='Probability')
print(f'class = {preds_class}')
print(f'proba = {preds_proba}')

# 特征重要性
print(f'feature_importance = {model.get_feature_importance(X_train, y_train)}')

# 保存模型
model.save_model('./boost/CatBoost/class.cbm', format="cbm")

# 获取模型参数
print(f'args config: {model.get_params()}')

# Calculate the Accuracy metric for the objects in the given dataset
print(f'score: {model.score(X_test, y_test)}')

# other metrics
acc = accuracy_score(y_test, preds_class)
print(f'acc is: {acc}')

# Get predicted RawFormulaVal
preds_raw = model.predict(X_test, prediction_type='RawFormulaVal')
print(preds_raw)
