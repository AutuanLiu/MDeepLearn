#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：SVR
   Description :  支持向量回归
   Email : autuanliu@163.com
   Date：2017/12/21
"""

from sklearn.datasets import load_diabetes
# from sklearn.svm import SVR
from sklearn.svm import LinearSVR

X, y = load_diabetes(return_X_y=True)

# 创建模型
# lreg = SVR(kernel='linear')
lreg = LinearSVR()

# 训练模型
lreg.fit(X, y)

# 获取模型参数
coef, inter = lreg.coef_, lreg.intercept_

# 输出结果
print('coef: {}\ninter {}'.format(coef, inter))
