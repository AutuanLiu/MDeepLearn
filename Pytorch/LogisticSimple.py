#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：LogisticSimple
   Description :  实现简单逻辑回归
   http://pytorch.org/docs/0.3.0/nn.html?highlight=seq#torch.nn.Sequential
   http://pytorch.org/docs/0.3.0/nn.html?highlight=add_module#torch.nn.Module.add_module
   Email : autuanliu@163.com
   Date：2017/12/16
"""

import numpy as np
import torch
from sklearn.datasets import load_iris
from torch import nn, optim
from torch.autograd import Variable

X, y = load_iris(return_X_y=True)

# 将 y 统一为矩阵的形式
X, y = X[:100], y[:100, np.newaxis]

# 为了结果的复现，设置种子
seed = 5
np.random.seed(seed)

# 分割数据集
train_index = np.random.choice(len(X), round(len(X) * 0.7), replace=False)
test_index = np.array(list(set(range(X.shape[0])) - set(train_index)))
train_X, train_y = X[train_index], y[train_index]
test_X, test_y = X[test_index], y[test_index]

# wrapper
train_X1 = Variable(torch.from_numpy(train_X).type(torch.FloatTensor))
train_y1 = Variable(torch.from_numpy(train_y).type(torch.FloatTensor))
test_X1 = Variable(torch.from_numpy(test_X).type(torch.FloatTensor))
test_y1 = Variable(torch.from_numpy(test_y).type(torch.FloatTensor))

# 构建模型
# model = nn.Sequential(
#     nn.Linear(4, 1),
#     nn.Sigmoid()
# )

# or using Sequential with add_module
# model = nn.Sequential()
# model.add_module('linear1', nn.Linear(4, 1))
# model.add_module('activation', nn.Sigmoid())

# 使用 OrderedDict
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('linear1', nn.Linear(4, 1)),
    ('activation', nn.Sigmoid())
])
)

# 损失函数，优化器
criterion = nn.BCELoss()
optimizer = optim.ASGD(model.parameters(), lr=0.01)

# train, 不使用 mini-batch
for epoch in range(500):
    optimizer.zero_grad()
    y_prediction = model(train_X1)
    loss = criterion(y_prediction, train_y1)
    loss.backward()
    optimizer.step()
    print('train epoch {} loss {}'.format(epoch + 1, loss.data[0]))

# test, 或许这里有不合理的地方
test_prediction = model(test_X1)
loss2 = torch.nn.functional.mse_loss(test_prediction, test_y1)
print('test loss {}'.format(loss2.data[0]))
