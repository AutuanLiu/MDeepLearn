#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：KernelRidgeRegression
   Description :  实现核岭回归
   使用随机数据, 中小规模速度比较快，大规模数据速度慢
   http://sklearn.apachecn.org/cn/0.19.0/auto_examples/plot_kernel_ridge_regression.html#sphx-glr-auto-examples-plot-kernel-ridge-regression-py
   Email : autuanliu@163.com
   Date：2017/12/18
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

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
krr_config = {'alpha': [1e0, 1e1, 1e2, 1e3],
              'gamma': np.logspace(-2, 2, num=5)
              }

# 实例化
krr = KernelRidge(kernel='rbf', gamma=0.1)

# 交叉验证
krr_cv = GridSearchCV(krr, param_grid=krr_config, cv=5)

# 拟合
krr_cv.fit(X, y)

# 预测
x1 = np.linspace(0, 5, 1000)[:, np.newaxis]
y_pred = krr_cv.predict(x1)

# 可视化
plt.scatter(X, y, alpha=0.5, label='sample')
plt.plot(x1, y_pred, 'r--', label='prediction curve', linewidth=2.)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Kernel Ridge Regression')
plt.legend(loc='best')
plt.show()
