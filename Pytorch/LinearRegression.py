#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：LinearRegression
   Description :  使用 sklearn 的 iris 数据，x: 花瓣宽度 y: 花瓣长度， 大致满足 线性关系
   Email : autuanliu@163.com
   Date：2017/12/15
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from torch.autograd import Variable

iris = load_iris()

# 将 y 统一为矩阵的形式
X = np.array([[d[3]] for d in iris.data])
y = np.array([[d[0]] for d in iris.data])

# 分割数据集
# 为了结果的复现，设置种子
seed = 1
np.random.seed(seed)

# train:test = 80%:20%
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(X.shape[0])) - set(train_index)))

# 获取数据集
train_X, train_y = X[train_index], y[train_index]
test_X, test_y = X[test_index], y[test_index]


# 构建模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear1(x)


# 参数设置
learning_rate = 0.1
epoch_num = 500
dtype = torch.FloatTensor


def main():
    # 模型实例
    model = MyModel()
    model.train()

    # 损失函数，优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)

    train_X1 = Variable(torch.from_numpy(train_X).type(dtype))
    train_y1 = Variable(torch.from_numpy(train_y).type(dtype))

    # train, 不使用 mini-batch
    for epoch in range(epoch_num):
        optimizer.zero_grad()
        y_prediction = model(train_X1)
        loss = criterion(y_prediction, train_y1)
        loss.backward()
        optimizer.step()
        # result
        print('train epoch {} loss {}'.format(epoch + 1, loss.data[0]))

    for name, para in model.named_parameters():
        print(name, para.data)

    # weight, bias 的另一种访问方式
    # print(model.linear1.weight.data[0, 0], model.linear1.bias.data[0])
    # test, 设置mode，只在 Dropout or BatchNorm 有影响，不设也可以
    model.eval()
    test_X1 = Variable(torch.from_numpy(test_X).type(dtype))
    test_y1 = Variable(torch.from_numpy(test_y).type(dtype))
    test_prediction = model(test_X1)
    loss2 = F.mse_loss(test_prediction, test_y1)
    print('test loss {}'.format(loss2.data[0]))


if __name__ == '__main__':
    main()
