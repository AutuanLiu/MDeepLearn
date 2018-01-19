#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：softmaxMnist
   Description : mnist data sets, softmax model
   pytorch 不需要进行 one-hot 编码, 使用类别即可
   Email : autuanliu@163.com
   Date：18-1-16
"""

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Module, functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_data(flag=True):
    mnist = MNIST(
        '../dataset/mnist/',
        train=flag,
        transform=transforms.ToTensor(),
        download=flag)
    loader = DataLoader(
        mnist, batch_size=config['batch_size'], shuffle=flag, drop_last=False)
    return loader


# 网络模型定义
class Network(Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(config['in_feature'], 500)
        self.l2 = nn.Linear(500, 350)
        self.l3 = nn.Linear(350, 200)
        self.l4 = nn.Linear(200, 130)
        self.l5 = nn.Linear(130, 10)

    def forward(self, x):
        data = x.view(-1, config['in_feature'])
        y = F.relu(self.l1(data))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        y = F.relu(self.l4(y))
        return self.l5(y)


def train_m(model, data_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion.forward(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{:5d}/{} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))


def test_m(model, data_loader):
    model.eval()
    test_loss, correct = 0, 0
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    len1 = len(data_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len1, 100. * correct / len1))


if __name__ == '__main__':
    # some config
    config = {
        'batch_size': 64,
        'epoch_num': 100,
        'lr': 0.001,
        'in_feature': 28 * 28,
        'out_feature': 10
    }
    train_loader, test_loader = get_data(), get_data(flag=False)
    # 模型实例与损失函数, 优化函数
    model = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    # 训练与测试
    for epoch in range(1, config['epoch_num']):
        train_m(model, train_loader, epoch)
    test_m(model, test_loader)
