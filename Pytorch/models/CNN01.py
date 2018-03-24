#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name：CNN01
    Description :  CNN 的简单实现  
    Email : autuanliu@163.com
    Date：2018/3/24
"""
import torch
import torch.nn as nn
use_gpu = torch.cuda.is_available()    # GPU


class SimpleCNN(nn.modules):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


def simpleCNN():
    return SimpleCNN().cuda() if use_gpu else SimpleCNN()
