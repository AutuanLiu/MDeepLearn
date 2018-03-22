#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   File Name：getdata
   Description : 用于训练与模型评估
   暂时只考虑CPU(比较穷没有GPU)
   GPU 没有获得测试
   Email : autuanliu@163.com
   Date：18-1-21
"""

import numpy as np
import torch

torch.manual_seed(5)

# from utils import gpu_t
from gpu import gpu_t


# train function and test function
def train_m(mod, train_data, opt, loss_f):
    """
        训练模型并返回本epoch的loss
    """
    mod.train()
    loss_epoch = []
    for batch_idx, (data, target) in enumerate(train_data):
        mod1, data1, target1 = gpu_t(mod, data, target)
        # 优化器清 0 操作(清除原先的grad)
        opt.zero_grad()
        y_pred = mod1.forward(data1)
        # loss 的前向与后向传播
        loss = loss_f.forward(y_pred, target1)
        loss.backward()
        # 更新模型的参数
        opt.step()
        # result
        loss_epoch.append(loss.data[0])
    return np.mean(loss_epoch)


def test_m(mod, test_data, loss_f):
    """
        用于对模型进行评估并返回 Average loss 和 Accuracy
    """
    mod.eval()
    test_loss, correct = 0, 0
    for data, target in test_data:
        mod1, data1, target1 = gpu_t(mod, data, target)
        output = mod1(data1)
        # sum up batch loss
        test_loss += loss_f(output, target1).data[0]
        # get the index of the max
        _, pred = output.data.max(1, keepdim=True)
        correct += pred.eq(target1.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_data.dataset)
    len1 = len(test_data.dataset)
    return test_loss, 100. * correct / len1
