#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   File Name：CNNfashionMnist
   Description : fashion mnist
   CNN
   Email : autuanliu@163.com
   Date：18-1-21
"""

import matplotlib.pyplot as plt
from getdata import get_fashionMnist
from torch import nn, optim
from torch.nn import Module, functional as F
from tqdm import tqdm
from train_eval import test_m, train_m


# 网络结构定义
class Network(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        y = F.relu(self.mp(self.conv1(x)))
        y = F.relu(self.mp(self.conv2(y)))
        y = y.view(in_size, -1)  # flatten the tensor
        y = self.fc(y)
        return F.log_softmax(y, dim=1)


if __name__ == '__main__':
    # 配置
    config = {
        'batch_size': 64,
        'epoch_num': 400,
        'lr': 0.01,
    }
    
    # 获取数据
    train_loader = get_fashionMnist(flag=True, batch=config['batch_size'])
    test_loader = get_fashionMnist(flag=False, batch=config['batch_size'])
    
    # criterion, optimizer define
    model = Network()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    # 训练与测试
    loss_trace = []
    ppar = tqdm(range(config['epoch_num']), desc='train progressbar: ')
    for epoch in ppar:
        epoch_loss = train_m(model, train_loader, optimizer, criterion)
        loss_trace.append(epoch_loss)
        ppar.set_postfix_str('current epoch loss: {:.5f}'.format(epoch_loss))
    print('loss: {}'.format(loss_trace))
    
    avg_loss, acc = test_m(model, test_loader, criterion)
    print('Average loss: {} Accuracy: {}'.format(avg_loss, acc))
    
    # 损失
    fig, ax = plt.subplots()
    ax.plot(loss_trace, label='loss curve')
    plt.show()
