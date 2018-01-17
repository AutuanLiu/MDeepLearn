#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：Logistic
   Description :  逻辑回归分类
   Email : autuanliu@163.com
   Date：2018/1/3
"""

import numpy as np
from collections import OrderedDict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F

from utils import gpu

X, y = load_iris(return_X_y=True)

# 将 y 统一为矩阵的形式
X, y = X[:100], y[:100, np.newaxis]

# 分割数据集
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=5)

# 使用 OrderedDict 构建模型
model = nn.Sequential(
    OrderedDict([('linear1', nn.Linear(4, 1)), ('activation', nn.Sigmoid())]))

# GPU 支持
model1, train_X1, train_y1 = gpu(model, train_X, train_y)
_, test_X1, test_y1 = gpu(model, test_X, test_y)

# 损失函数，优化器
criterion = nn.BCELoss()
optimizer = optim.ASGD(model1.parameters(), lr=0.01)

# train, 不使用 mini-batch
for epoch in range(500):
    optimizer.zero_grad()
    y_prediction = model1(train_X1)
    loss = criterion(y_prediction, train_y1)
    loss.backward()
    optimizer.step()
    print('train epoch {} loss {}'.format(epoch + 1, loss.data[0]))

# test, 或许这里有不合理的地方
test_prediction = model1(test_X1)
loss2 = F.mse_loss(test_prediction, test_y1)
print('test loss {}'.format(loss2.data[0]))
