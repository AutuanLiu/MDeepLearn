#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：RandomForest
   Description :
   1. 随机森林
   2. 决策树
   3. AdaBoost 分类
   4. GradientBoosting 分类
   5. extraTrees 分类
   6. 高斯过程
   Email : autuanliu@163.com
   Date：2017/12/24
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)

dtc = DecisionTreeClassifier()
# 弱学习器的数量由参数 n_estimators 来控制
rfc = RandomForestClassifier(n_estimators=10, criterion='gini')
etc = ExtraTreesClassifier(n_estimators=10)
abc = AdaBoostClassifier(n_estimators=50)
gbc = GradientBoostingClassifier(n_estimators=100)
vcf = VotingClassifier(estimators=[('rfc', rfc), ('etc', etc), ('gbc', gbc)], voting='hard')
gpc = GaussianProcessClassifier()

dtc.fit(X, y)
rfc.fit(X, y)
etc.fit(X, y)
abc.fit(X, y)
gbc.fit(X, y)
vcf.fit(X, y)
gpc.fit(X, y)

scores = cross_val_score(dtc, X, y)
scores1 = cross_val_score(rfc, X, y)
scores2 = cross_val_score(etc, X, y)
scores3 = cross_val_score(abc, X, y).mean()
scores4 = cross_val_score(gbc, X, y).mean()
scores5 = cross_val_score(vcf, X, y).mean()
scores6 = cross_val_score(gpc, X, y).mean()

# 预测测试, 对应的标签是 0, 1, 2, 1
test = [
    [4.0, 3.1, 1.1, 0.1],
    [6.7, 3.1, 4.1, 1.4],
    [7.1, 3.2, 6.1, 1.9],
    [6.3, 2.9, 4.2, 1.4],
]

y_pred0 = dtc.predict(test)
y_pred = rfc.predict(test)
y_pred1 = etc.predict(test)
y_pred2 = abc.predict(test)
y_pred3 = gbc.predict(test)
y_cvf = vcf.predict(test)
y_gpc = gpc.predict(test)

print(y_pred0, y_pred, y_pred1, y_pred2, y_pred3)
print(scores.mean(), scores1.mean(), scores2.mean(), scores3, scores4, abc.feature_importances_)
print(y_cvf, scores5, y_gpc, scores6)
