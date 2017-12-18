#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：SupportVectorRegression
   Description :  
   Email : autuanliu@163.com
   Date：2017/12/18
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# 数据生成
rng = np.random.RandomState(0)
dim1, dim2, step = 700, 1, 5

# 放大数字[0, 5)
X = rng.rand(dim1, dim2) * 5

# 返回一个拉平的数组
y = np.sin(X).ravel()

# 增加噪声, 每个step个数据点加噪声，任意的
# y + [-1.5, 1.5), 上下偏移 1.5
y[::step] += 3 * (0.5 - rng.rand(dim1 // step))

# fit
svr_config = {'C': [1e0, 1e1, 1e2, 1e3],
              'gamma': np.logspace(-2, 2, num=5)
              }

# 实例化
svr = SVR(kernel='rbf', gamma=0.1)

# 交叉验证
svr_cv = GridSearchCV(svr, param_grid=svr_config, cv=5)

# 拟合
svr_cv.fit(X, y)

# 预测
x1 = np.linspace(0, 5, 1000)[:, np.newaxis]
y_pred = svr_cv.predict(x1)

# 可视化
plt.scatter(X, y, alpha=0.5, label='sample')
plt.plot(x1, y_pred, 'r--', label='prediction curve', linewidth=2.)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Support Vector Regression')
plt.legend(loc='best')
plt.show()
