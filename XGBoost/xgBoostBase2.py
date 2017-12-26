#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：xgBoostBase2
   Description :  使用 XGBoost 进行回归任务
   主要使用 sklearn wrapper
   Email : autuanliu@163.com
   Date：2017/12/26
"""

import xgboost as xgb
from sklearn.datasets import load_boston

# 导入数据
X, y = load_boston(return_X_y=True)

# 训练模型
xgbc = xgb.XGBClassifier().fit(X, y)

# 预测测试, 对应的结果 [22.4]
test = [[6.26300000e-02, 0.00000000e+00, 1.19300000e+01,
         0.00000000e+00, 5.73000000e-01, 6.59300000e+00,
         6.91000000e+01, 2.47860000e+00, 1.00000000e+00,
         2.73000000e+02, 2.10000000e+01, 3.91990000e+02,
         9.67000000e+00]
        ]

# 预测
y_pred = xgbc.predict(test)

print(y_pred)
