#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：SGD
   Description :  
   Email : autuanliu@163.com
   Date：2017/12/22
"""

# SGD 分类器
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

X = [[0, 0], [1, 1]]
y = [0, 1]

clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=500, tol=1e-3)
clf.fit(X, y)
# prediction
res = clf.predict([[2, 2]])
print(res, clf.coef_, clf.intercept_)

# 决策树分类
# 可以预测类别， 也可以预测 每个类的概率
dtc = DecisionTreeClassifier(criterion='gini', max_depth=2)
dtc.fit(X, y)
y_pred = dtc.predict([[2, 2]])
y_p = dtc.predict_proba([[2, 2]])
print(y_pred, y_p)
