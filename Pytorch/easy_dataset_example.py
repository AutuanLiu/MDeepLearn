#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   File Name：easy_dataset_example
   Description :  
   Email : autuanliu@163.com
   Date：18-1-22
"""

from collections import OrderedDict

import numpy as np
import torch
from make_dataset import DataDef
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Sequential
from torch.utils.data import DataLoader

torch.manual_seed(4)
# 获取数据
iris = DataDef('../datasets/iris.csv', [0, 150], [0, 4], [4, 5])
train_loader = DataLoader(dataset=iris, batch_size=8, shuffle=True, num_workers=2)

# 模型
model = Sequential(OrderedDict([('linear1', nn.Linear(4, 1)), ('activation', nn.Sigmoid())]))

# 损失函数，优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(50):
    loss_epoch = []
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        y_pred = model.forward(inputs)
        # loss 的前向与后向传播
        loss = criterion.forward(y_pred, labels)
        loss.backward()
        # 更新模型的参数
        optimizer.step()
        # Run your training process
        loss_epoch.append(loss.data[0])
    print(epoch + 1, "loss: ", np.mean(loss_epoch))
