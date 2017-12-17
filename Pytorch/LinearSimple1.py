#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：LinearSimple1
   Description :  更简单的方式实现线性回归
   Email : autuanliu@163.com
   Date：2017/12/17
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 加载数据
iris = load_iris()
data = np.array([[d[3]] for d in iris.data])
target = np.array([[d[0]] for d in iris.data])

# 参数设置
learning_rate = 0.01
epoch_num = 500
dtype = torch.FloatTensor

# 创建模型
model = nn.Sequential(nn.Linear(1, 1))

# wrapper
data_train = Variable(torch.from_numpy(data).type(dtype))
target_train = Variable(torch.from_numpy(target).type(dtype))

# criterion, optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

# 开始训练模型
loss_trace = []
for epoch in range(epoch_num):
    y_pred = model(data_train)
    optimizer.zero_grad()
    loss = criterion(y_pred, target_train)
    loss.backward()
    optimizer.step()
    # result
    loss_trace.append(loss.data[0])
    print('epoch {}, loss {}'.format(epoch + 1, loss.data[0]))

# 获取各个参数 weight, bias
weight = model[0].weight.data[0, 0]
bias = model[0].bias.data[0]

# 可视化
plt.plot(data, weight * data + bias, label='prediction')
plt.plot(data, target, 'o', label='origin')
plt.legend(loc='best')
plt.show()

plt.plot(loss_trace)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
