#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：softmaxMnist
   Description :  fashion mnist and mnist data sets, softmax model
   Email : autuanliu@163.com
   Date：18-1-16
"""

import torch.nn as nn
from torch.nn import Module, functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST

train = DataLoader(FashionMNIST('./dataset'), batch_size=64)
train1 = DataLoader(MNIST('../dataset'), batch_size=64)
print(FashionMNIST('../dataset', download=True))


class Network(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 2)
        pass

    def forward(self, x):
        y = F.softmax(x)
        pass
