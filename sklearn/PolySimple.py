#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：PolySimple
   Description :  多项式回归
   数据使用随机生成的数据
   http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#polynomial-regression
   Email : autuanliu@163.com
   Date：2017/12/17
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# 使用1个特征, 500 个样本
x = np.arange(500).reshape(500, 1)
y = 5 + x + 2 * x ** 2 + 3 * x ** 3

# pipeline形式
model = Pipeline([('ploy', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))
                  ])

# fit
model.fit(x, y)

# result
print('coef: {}'.format(model.named_steps['linear'].coef_))
