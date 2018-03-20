#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：RidgeRegression
   Description :  岭回归（带有超参数的交叉验证）
   Email : autuanliu@163.com
   Date：2017/12/14
"""

# 用于 lasso回归
# from sklearn.linear_model import Lasso
# 用于贝叶斯岭回归
# from sklearn.linear_model import BayesianRidge
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

# 将 y 统一为矩阵的形式
y = y[:, np.newaxis]

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

# 设置需要交叉验证的超参数
# alpha = [0.1, 1.0, 10.0]
alpha = list(np.arange(0.01, 0.5, 0.01))

# 创建模型
lreg = RidgeCV(alpha)
# lreg = Lasso(0.1)
# 贝叶斯岭回归，速度比较慢
# lreg = BayesianRidge()

# 训练模型
lreg.fit(train_X, train_y)

# 测试模型
test_pred = lreg.predict(test_X)

# 获取模型参数
coef, inter, alp = lreg.coef_, lreg.intercept_, lreg.alpha_

# 评价模型
mse = mean_squared_error(test_y, test_pred)
r2 = r2_score(test_y, test_pred)

# 输出结果
print('coef: {}\ninter {}\nalpha {}\nmse {}\nr2 {}'.format(coef, inter, alp, mse, r2))
