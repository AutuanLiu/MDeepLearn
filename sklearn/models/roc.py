# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：roc
   Description :  ROC AUC
   Email : autuanliu@163.com
   Date：2018/1/2
"""

from sklearn.datasets import load_iris

# data, 2 class
X, y = load_iris(return_X_y=True)
X, y = X[y != 2], y[y != 2]

print(X, y)
