#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：class
   Description :  lightGBM 回归实例 with early-stopping
   Email : autuanliu@163.com
   Date：2018/3/20
"""
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 获取数据
data, target = make_regression(
    n_samples=1000, n_features=10, n_targets=1, n_informative=8, noise=0.1, random_state=12, bias=1.2)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# lgb 格式数据
train = lgb.Dataset(X_train, label=y_train)
test = lgb.Dataset(X_test, label=y_test, reference=train)

# 参数设置
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.01, 
    'num_iterations': 100,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# train
# record eval results for plotting
res = {}
gbm = lgb.train(params, train, valid_sets=test, early_stopping_rounds=5, evals_result=res)

# save model to file
gbm.save_model('./boost/LightGBM/regression.txt')

# predict
print(f'Best iterations: {gbm.best_iteration}')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# plot
ax = lgb.plot_metric(res, metric='auc')
plt.show()
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()

