#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：NaiveBayes
   Description :  朴素贝叶斯
   Email : autuanliu@163.com
   Date：2017/12/22
"""

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

X, y = datasets.load_iris(return_X_y=True)

# 高斯朴素贝叶斯
gnb = GaussianNB()

# 拟合预测
gnb.fit(X, y)
y_pred = gnb.predict(X)
print(y_pred)

# 错误个数
cnt = (y != y_pred)
print('错误个数 {}'.format(cnt.sum()))
