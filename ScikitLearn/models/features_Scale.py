#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：feature
   Description :  特征提取和预处理
   Email : autuanliu@163.com
   Date：18-1-12
"""

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import (scale, StandardScaler, MaxAbsScaler, MinMaxScaler, normalize, Normalizer, Binarizer)

measurements = [{
    'city': 'Dubai',
    'temperature': 33.
}, {
    'city': 'London',
    'temperature': 12.
}, {
    'city': 'San Francisco',
    'temperature': 18.
}]

vec = DictVectorizer()

vec.fit_transform(measurements).toarray()

names = vec.get_feature_names()
print(names)

# preprocessing
x = np.array([[-1, 2, 1.3], [3.4, -2.1, 4.2], [-3, 2.1, 5.1]])

# scale 快速标准化
scaler = scale(x)
print('scale:\n', scaler, scaler.mean(axis=0), scaler.std(axis=0), '\n')

# StandardScaler
scaler1 = StandardScaler().fit(x)
x_trans = scaler1.transform(x)
print('StandardScaler:\n', x_trans, scaler1.mean_, scaler1.scale_, '\n')

# MinMaxScaler
scaler2 = MinMaxScaler().fit(x)
x_trans1 = scaler2.transform(x)
print('MinMaxScaler:\n', x_trans1, scaler2.min_, scaler2.scale_, '\n')

# MaxAbsScaler
scaler3 = MaxAbsScaler().fit(x)
x_trans2 = scaler3.transform(x)
print('MaxAbsScaler:\n', x_trans2, scaler3.scale_, '\n')

# normalize
x_trans3 = normalize(x, norm='l2', axis=0, return_norm=True)
print('normalize:\n', x_trans3)
nor = Normalizer(norm='l2').fit(x)
x_trans4 = nor.transform(x)
print('Normalizer:\n', x_trans4)

# Binarizer
nor1 = Binarizer(threshold=0.2).fit(x)
x_trans5 = nor1.transform(x)
print('Binarizer:\n', x_trans5)
