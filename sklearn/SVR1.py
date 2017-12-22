#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：SVR1
   Description :
   http://sklearn.apachecn.org/cn/0.19.0/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
   Email : autuanliu@163.com
   Date：2017/12/21
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVR

# s生成数据
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# 加噪音
y[::5] += 3 * (0.5 - np.random.rand(8))

# 拟合模型
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_rbf.fit(X, y), svr_lin.fit(X, y), svr_poly.fit(X, y)
y_rbf = svr_rbf.predict(X)
y_lin = svr_lin.predict(X)
y_poly = svr_poly.predict(X)

# 结果
sns.set(style='whitegrid')
colors = sns.color_palette("Set2", 3)
names = ['RBF model', 'Linear model', 'Polynomial model']
plt.figure(1)
plt.scatter(X, y, color='orange', label='data')
for data_y, color_i, name in zip([y_rbf, y_lin, y_poly], colors, names):
    plt.plot(X, data_y, color=color_i, label=name)
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend(loc='best')
plt.show()
