#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：LeastSquaresRegression
   Description :  最小二乘回归
   Email : autuanliu@163.com
   Date：2017/12/14
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
# 分割数据集
# 为了结果的复现，设置种子
seed = 1
np.random.seed(seed)

# X.shape = (442, 10), y.shape = (442,)
# train:test = 80%:20%
# 不放回抽样
# np.round() 返回浮点数，而round()返回int型
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(X.shape[0])) - set(train_index)))

# 获取数据集
train_X, train_y = X[train_index], y[train_index]
test_X, test_y = X[test_index], y[test_index]

# 创建模型
lreg = LinearRegression()

# 训练模型
lreg.fit(train_X, train_y)

# 测试模型
test_pred = lreg.predict(test_X)

# 获取模型参数
coef, inter = lreg.coef_, lreg.intercept_

# 评价模型
mse = mean_squared_error(test_y, test_pred)
r2 = r2_score(test_y, test_pred)

# 输出结果
print('coef: {}\ninter {}\nmse {}\nr2 {}'.format(coef, inter, mse, r2))

