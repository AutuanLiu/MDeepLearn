#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：basic01
   Description :  catboost 实例
   Email : autuanliu@163.com
   Date：2018/3/18
"""
from catboost import CatBoostRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 加载数据
X, y = load_diabetes(return_X_y=True)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 训练数据
clf = CatBoostRegressor(iterations=800, learning_rate=0.8, depth=6, loss_function='RMSE')
fit_model = clf.fit(X_train, y_train)

# 模型参数
print(fit_model.get_params())

# 预测模型
y_pred = clf.predict(X_test)

# 评估模型
print('mean squared error: {}'.format(mean_squared_error(y_test, y_pred)))
print('r2 score: {}'.format(r2_score(y_test, y_pred)))
