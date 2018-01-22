#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   File Name：getdata
   Description : 用于获取数据集并返回 dataloader
   Email : autuanliu@163.com
   Date：18-1-21
"""

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST


# 获取数据
def get_fashionMnist(flag=True, batch=64):
    """
        获取 fashionMNIST 数据集
        num_workers 根据电脑情况设置
    """
    fashionMnist = FashionMNIST(
        '../datasets/fashionMnist/',
        train=flag,
        transform=transforms.ToTensor(),
        download=flag)
    
    loader = DataLoader(
        fashionMnist, batch_size=batch, shuffle=flag, drop_last=False, num_workers=8)
    return loader


def get_Mnist(flag=True, batch=64):
    """
        获取 MNIST 数据集
    """
    mnist = MNIST(
        '../datasets/Mnist/',
        train=flag,
        transform=transforms.ToTensor(),
        download=flag)
    
    loader = DataLoader(
        mnist, batch_size=batch, shuffle=flag, drop_last=False, num_workers=8)
    return loader
