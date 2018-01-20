#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：logSoftmax
   Description : fashion mnist
   Softmax + CrossEntropy = logSoftmax + NLLLoss
   http://willwolf.io/2017/05/18/minimizing_the_negative_log_likelihood_in_english/
   Email : autuanliu@163.com
   Date：18-1-19
"""

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST


def get_data(flag=True):
    mnist = FashionMNIST(
        '../dataset/fashionMnist/',
        train=flag,
        transform=transforms.ToTensor(),
        download=flag)
    loader = DataLoader(
        mnist, batch_size=config['batch_size'], shuffle=flag, drop_last=False)
    return loader


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