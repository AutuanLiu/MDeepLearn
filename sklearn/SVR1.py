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
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

print(__doc__)

# 生成数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# 加噪音
y[::5] += 3 * (0.5 - np.random.rand(8))

# 拟合模型
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
knng = KNeighborsRegressor(n_neighbors=6, weights='uniform')
rng = RadiusNeighborsRegressor(radius=1.0, weights='uniform')
dtr = DecisionTreeRegressor(criterion='mse')
abr = AdaBoostRegressor(n_estimators=50)
rfr = RandomForestRegressor(n_estimators=50)

svr_rbf.fit(X, y), svr_lin.fit(X, y), svr_poly.fit(X, y)
knng.fit(X, y), rng.fit(X, y), dtr.fit(X, y)
abr.fit(X, y), rfr.fit(X, y)

# 支持向量回归
y_rbf = svr_rbf.predict(X)
y_lin = svr_lin.predict(X)
y_poly = svr_poly.predict(X)

# KNN 回归
y_knng = knng.predict(X)
y_rng = rng.predict(X)

# 决策树回归
y_dtr = dtr.predict(X)

# ensemble
y_abr = abr.predict(X)
y_rfr = rfr.predict(X)

# 结果
sns.set(style='whitegrid')
colors = sns.color_palette('Set2', 8)
names = ['RBF model', 'Linear model', 'Polynomial model', 'KNR', 'RNR', 'DTR', 'ABR', 'RFR']
data_pred = [y_rbf, y_lin, y_poly, y_knng, y_rng, y_dtr, y_abr, y_rfr]
plt.figure(1)
plt.scatter(X, y, color='red', label='data')
for data_y, color_i, name in zip(data_pred, colors, names):
    plt.plot(X, data_y, color=color_i, label=name)
plt.xlabel('data')
plt.ylabel('target')
plt.legend(loc='best')
plt.show()
