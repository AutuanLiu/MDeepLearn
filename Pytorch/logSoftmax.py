#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：logSoftmax
   Description : fashion mnist
   Softmax + CrossEntropy = logSoftmax + NLLLoss
   ref1: http://willwolf.io/2017/05/18/minimizing_the_negative_log_likelihood_in_english/
   顺便加入 tensorBoard 的可视化操作
   ref2: http://tensorboard-pytorch.readthedocs.io/en/latest/tensorboard.html
   ref3: https://github.com/lanpa/tensorboard-pytorch
   Email : autuanliu@163.com
   Date：18-1-19
"""

import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Module, Sequential, functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from utils.logger import Logger


def get_data(flag=True):
    mnist = FashionMNIST(
        '../dataset/fashionMnist/',
        train=flag,
        transform=transforms.ToTensor(),
        download=flag)
    loader = DataLoader(
        mnist, batch_size=config['batch_size'], shuffle=flag, drop_last=False)
    return loader


# 模型结构
class Network(Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(config['in_feature'], 500)
        self.l2 = nn.Linear(500, 350)
        self.l3 = nn.Linear(350, 200)
        self.l4 = nn.Linear(200, 130)
        self.log1 = Sequential(
            nn.Linear(130, config['out_feature']), nn.LogSoftmax(dim=1))

    def forward(self, data):
        # 数据展开
        x = data.view(-1, config['in_feature'])
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        y = F.relu(self.l4(y))
        y = self.log1(y)
        return y


# train function and test function
def train_m(mod, train_data):
    mod.train()
    loss_epoch = []
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        y_pred = mod.forward(data)
        loss = criterion.forward(y_pred, target)
        loss.backward()
        optimizer.step()
        # result
        if batch_idx % 10 == 0:
            print('epoch: {} [{:5.0f}/{} {:2.0f}%] loss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_data.dataset),
                100 * batch_idx / len(train_data), loss.data[0]))
            loss_epoch.append(loss.data[0])

            # loss log
            logger.scalar_summary('loss', np.mean(loss_epoch), epoch + 1)

            # Log values and gradients of the parameters (histogram)
            for tag, value in mod.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

            # Log the images
            info = {
                'images': (data.view(-1, 28, 28)[:10]).data.cpu().numpy()
            }

            for tag, images in info.items():
                logger.image_summary(tag, images, epoch + 1)


def test_m(mod, test_data):
    mod.eval()
    test_loss, correct = 0, 0
    for data, target in test_data:
        data, target = Variable(data, volatile=True), Variable(target)
        output = mod(data)
        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max
        _, pred = output.data.max(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_data.dataset)
    len1 = len(test_data.dataset)
    print(
        'Test set: \nAverage loss: {:.4f}, Accuracy: {}/{} ({}%)'.format(
            test_loss, correct, len1, 100. * correct / len1))


# some config
config = {
    'batch_size': 64,
    'epoch_num': 200,
    'lr': 0.01,
    'in_feature': 28 * 28,
    'out_feature': 10
}

# log record
logger = Logger('./logs')
train_loader, test_loader = get_data(), get_data(flag=False)

# criterion, optimizer define
model = Network()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=config['lr'])

# 训练与测试
for epoch in range(config['epoch_num']):
    train_m(model, train_loader)
test_m(model, test_loader)
