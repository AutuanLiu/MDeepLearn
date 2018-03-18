# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：xgboostBase3
   Description :  回归预测
   Email : autuanliu@163.com
   Date：2017/12/26
"""

import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 加载数据
X, y = load_diabetes(return_X_y=True)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 训练数据
clf = xgb.XGBRegressor()
clf.fit(X_train, y_train)

# 预测模型
y_pred = clf.predict(X_test)

# 评估模型
print('mean squared error: {}'.format(mean_squared_error(y_test, y_pred)))
print('mean absolute error: {}'.format(mean_absolute_error(y_test, y_pred)))
print('r2 score: {}'.format(r2_score(y_test, y_pred)))
