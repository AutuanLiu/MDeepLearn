#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：q
   Description : 二次判别分析
   生成二次判别边界，根据贝叶斯公式，也即后验概率来生成每个类的边界
   假设每个类都是服从 高斯分布
   预测：根据每个类的判别函数计算属于该类的后验概率，取概率最大者对应的类为预测结果
   详情参考：《模式分类》 Duda等著, 李宏东等译
   Email : autuanliu@163.com
   Date：2017/12/18
"""

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# 导入数据
X, y = load_iris(return_X_y=True)

# 实例化
qda = QuadraticDiscriminantAnalysis()

# 拟合
qda.fit(X, y)

# 预测测试, 对应的标签是 0, 1, 2, 1
test = [
    [4.0, 3.1, 1.1, 0.1],
    [6.7, 3.1, 4.1, 1.4],
    [7.1, 3.2, 6.1, 1.9],
    [6.3, 2.9, 4.2, 1.4],
]
prediction = qda.predict(test)
print('predict result: {}'.format(prediction))
