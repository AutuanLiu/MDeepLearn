#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：regression
   Description :  catboost 回归实例
   1. https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/#python-reference_parameters-list
   2. https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostregressor-docpage/
   Email : autuanliu@163.com
   Date：2018/3/19
"""
from catboost import Pool, CatBoostRegressor, cv
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data, target, coef = make_regression(n_samples=1000, n_features=10, n_targets=1, n_informative=8, noise=0.1, random_state=12, bias=1.2, coef=True)
print(f'real coef: {coef}')

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)
# 使用 CatBoost 格式的数据 Pool
train_pool, test_pool = Pool(X_train, y_train), Pool(X_test)

# 构建模型
config = {
    'iterations': 800,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'RMSE',
    'logging_level': 'Verbose',
    'random_seed': 120,
    'l2_leaf_reg': 2,
    'nan_mode': 'Min',
    'calc_feature_importance': True,
    'leaf_estimation_method': 'Gradient',
    'metric_period': 5,
    'train_dir': 'logs'
}

# unpacking 的形式传入参数
model = CatBoostRegressor(**config)

# 模型训练与预测
model.fit(train_pool)
y_pred = model.predict(test_pool)

# 模型评估
print(f'R2: {r2_score(y_test, y_pred)}')

# 特征重要性
print(f'feature_importance = {model.get_feature_importance(train_pool)}')

# 保存模型
model.save_model('regression.cbm', format="cbm")

# Calculate the RMSE metric for the objects in the given dataset.
print(f'score: {model.score(X_train, y_train)}')

# cv
args = config.pop('calc_feature_importance', None)
print(config)
config1 = {
    'pool': train_pool,
    'params': config,
    'iterations': 1000,
    'fold_count': 10,
    'partition_random_seed': 120,
    'logging_level': 'Verbose',
    'stratified': False,
    'as_pandas': True
}

scores = cv(**config1)
print(f'CV result is: {scores}')
