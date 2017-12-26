#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：featureSelection
   Description :  特征选择
   Email : autuanliu@163.com
   Date：2017/12/26
"""

from sklearn.feature_selection import VarianceThreshold

X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]

# 方差阈值 特征选择
# 移除那些在整个数据集中特征值为0或者为1的比例超过80%的特征
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
x_trans = sel.fit_transform(X)

print(x_trans)

# 使用 卡方检验 选择最好的两个特征
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2

X, y = load_iris(return_X_y=True)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)

print(X_new, X_new.shape)
