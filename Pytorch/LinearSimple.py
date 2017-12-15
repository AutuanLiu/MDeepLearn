#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：LinearSimple
   Description :  线性回归的精简版，同时支持自动判别CPU，GPU或多GPU并行
   使用 sklearn 的 iris 数据，x: 花瓣宽度 y: 花瓣长度， 大致满足 线性关系
   Email : autuanliu@163.com
   Date：2017/12/15
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 加载数据, 这里只使用一个特征 bp
iris = load_iris()
data = np.array([[d[3]] for d in iris.data])
target = np.array([[d[0]] for d in iris.data])

# 参数设置
learning_rate = 0.01
epoch_num = 500
dtype = torch.FloatTensor


# 构建模型结构
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear1(x)


# 创建模型
model = MyModel()

# GPU, CPU, 并行等处理
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model = model.cuda()
    data_train = Variable(torch.from_numpy(data).type(dtype).cuda())
    target_train = Variable(torch.from_numpy(target).type(dtype).cuda())
else:
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

# model.parameters()是一个生成器
param = []
# 获取各个参数 weight, bias
# https://discuss.pytorch.org/t/getting-only-weights-not-biases-from-a-module/1838
for _, para in model.named_parameters():
    param.append(para.data.numpy().squeeze())

# 可视化
plt.plot(data, param[0] * data + param[1], label='prediction')
plt.plot(data, target, 'o', label='origin')
plt.legend(loc='best')
plt.show()

plt.plot(loss_trace)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
