#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：curves
   Description :  学习曲线, 验证曲线
   Email : autuanliu@163.com
   Date：2018/1/2
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
train_sizes, train_scores, valid_scores = learning_curve(
    SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
print('train_sizes: {}\n'.format(train_sizes))
print('train_scores: {}\n'.format(train_scores))
print('valid_scores: {}\n'.format(valid_scores))

train_scores1, valid_scores1 = validation_curve(Ridge(), X, y, "alpha",
                                                np.logspace(-7, 3, 3))
print('train_scores1: {}\n'.format(train_scores1))
print('train_scores1: {}\n'.format(valid_scores1))
