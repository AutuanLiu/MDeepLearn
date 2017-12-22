#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：DecisionTree
   Description :  决策树分类
   Email : autuanliu@163.com
   Date：2017/12/22
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)

clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X, y)

y_pred = clf.predict(X[:1, :])
y_p = clf.predict_proba(X[:1, :])
print(y_pred, y_p)
