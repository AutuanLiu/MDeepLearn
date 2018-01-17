#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：SVM
   Description :  使用iris数据实现一个SVM的线性分类器
   Email : autuanliu@163.com
   Date：2017/12/10
"""

from sklearn.datasets import load_iris

# 数据获取
X, y = load_iris(return_X_y=True)
