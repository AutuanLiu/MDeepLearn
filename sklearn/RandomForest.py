#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：RandomForest
   Description :  使用随机森林预测红酒的质量
   数据：http://archive.ics.uci.edu/ml/datasets/Wine
   Email : autuanliu@163.com
   Date：2017/12/10
"""

from sklearn.datasets import load_wine

# 数据获取
X, y = load_wine(return_X_y=True)
