#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：class
   Description :  lightGBM 回归实例 with early-stopping sklearn API
   Email : autuanliu@163.com
   Date：2018/3/20
"""
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# 获取数据
data, target = make_regression(n_samples=1000, n_features=10, n_targets=1, n_informative=8, noise=0.1, random_state=12, bias=1.2)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# 模型构建
gbm = LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=500)

# 训练
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l2', early_stopping_rounds=5)

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
print(f'Best iterations: {gbm.best_iteration_}')

# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred)**0.5)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))

# or
estimator = LGBMRegressor(num_leaves=31)

param_grid = {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [300, 500, 700]}

# 网格搜索
gbm1 = GridSearchCV(estimator, param_grid)

# 拟合模型
gbm1.fit(X_train, y_train)

# 网格搜索的结果
print('Best parameters found by grid search are:', gbm1.best_params_)
