#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：xgBoostBase
   Description : 使用 XGBoost 做分类任务
   Email : autuanliu@163.com
   Date：2017/12/26
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris

# 导入数据
X, y = load_iris(return_X_y=True)

# 训练模型
xgbc = xgb.XGBClassifier().fit(X, y)

# 保存于加载模型
pickle.dump(xgbc, open("clf.pkl", "wb"))
clf2 = pickle.load(open("clf.pkl", "rb"))

# 预测测试, 对应的标签是 0, 1, 2, 1
test = [[4.0, 3.1, 1.1, 0.1],
        [6.7, 3.1, 4.1, 1.4],
        [7.1, 3.2, 6.1, 1.9],
        [6.3, 2.9, 4.2, 1.4],
        ]

# 预测
y_pred = xgbc.predict(test)
y_pred1 = clf2.predict(test)

print(y_pred, y_pred1)

# 是否在 公差范围内相等
print('equal status: {}'.format(np.allclose(y_pred, y_pred1)))

# 可视化
# 特征重要性
xgb.plot_importance(xgbc)
plt.show()

# xgb.plot_tree(xgbc)
