#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：RandomForest
   Description :  使用随机森林预测红酒的质量
   数据：http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
   Email : autuanliu@163.com
   Date：2017/12/10
"""

import pandas as pd

# 数据获取
data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data_src = pd.read_table(data_url, sep=';')
print(data_src.head())
