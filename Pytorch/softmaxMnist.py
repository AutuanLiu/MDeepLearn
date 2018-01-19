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
from torch.nn import Module, functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_data():
    mnist = MNIST(
        '../dataset/mnist/',
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor,
             transforms.Normalize(0, 1)]))
    train1 = DataLoader(MNIST('../dataset'), batch_size=64)


class Network(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 2)
        pass

    def forward(self, x):
        y = F.softmax(x)
        return y


if __name__ == '__main__':
    # some config
    batch_size = 64
    get_data()
