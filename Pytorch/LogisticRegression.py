#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：LosisticRegression
   Description :  实现逻辑回归(分类算法)
   逻辑回归可以将线性回归转换为一个二值(多值)分类器， 通过 sigmoid函数将线性回归的输出缩放到0~1之间，
   如果目标值在阈值之上则预测为1类, 否则预测为0类
   Email : autuanliu@163.com
   Date：2017/12/15
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from torch.autograd import Variable

X, y = load_iris(return_X_y=True)

# 将 y 统一为矩阵的形式
X = X[:100]
y = y[:100, np.newaxis]

# 设置参数
learning_rate = 0.01
epoch_num = 500
dtype = torch.FloatTensor
rate = 0.7
in_feature = 4
out_feature = 1

# 为了结果的复现，设置种子
seed = 1
np.random.seed(seed)

# 分割数据集
train_index = np.random.choice(len(X), round(len(X) * rate), replace=False)
test_index = np.array(list(set(range(X.shape[0])) - set(train_index)))
train_X, train_y = X[train_index], y[train_index]
test_X, test_y = X[test_index], y[test_index]


# 构建模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_feature, out_feature)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear1(x))
        return y_pred


# 模型实例
model = MyModel()

# CPU，GPU，并行支持，这里可以简单写，如果只是CPU，GPU等
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model = model.cuda()
    # wrapper
    train_X1 = Variable(torch.from_numpy(train_X).type(dtype).cuda())
    train_y1 = Variable(torch.from_numpy(train_y).type(dtype).cuda())
    test_X1 = Variable(torch.from_numpy(test_X).type(dtype).cuda())
    test_y1 = Variable(torch.from_numpy(test_y).type(dtype).cuda())
else:
    train_X1 = Variable(torch.from_numpy(train_X).type(dtype))
    train_y1 = Variable(torch.from_numpy(train_y).type(dtype))
    test_X1 = Variable(torch.from_numpy(test_X).type(dtype))
    test_y1 = Variable(torch.from_numpy(test_y).type(dtype))

# 损失函数，优化器
criterion = nn.BCELoss()
optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

# train, 不使用 mini-batch
loss_trace = []
for epoch in range(epoch_num):
    optimizer.zero_grad()
    y_prediction = model(train_X1)
    loss = criterion(y_prediction, train_y1)
    loss.backward()
    optimizer.step()
    # result
    loss_trace.append(loss.data[0])
    print('train epoch {} loss {}'.format(epoch + 1, loss.data[0]))

# test
test_prediction = model(test_X1)
loss2 = F.mse_loss(test_prediction, test_y1)
print('test loss {}'.format(loss2.data[0]))

plt.plot(loss_trace)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
