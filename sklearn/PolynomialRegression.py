#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：PolynomialRegression
   Description :  多项式回归
   数据使用随机生成的数据
   http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#polynomial-regression
   Email : autuanliu@163.com
   Date：2017/12/17
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 使用1个特征, 500 个样本
x = np.arange(500).reshape(500, 1)
y = 5 + x + x ** 2 + 3 * x ** 3

# 变换
ploy = PolynomialFeatures(degree=3)
# 2个特征 --> 6个特征
ploy_x = ploy.fit_transform(x)

# 拟合, 特征中的第一列包含了截距，这里要设 False
model = LinearRegression(fit_intercept=False)
model.fit(ploy_x, y)

# 结果与我们设置的一样
print('coef: {}'.format(model.coef_))
