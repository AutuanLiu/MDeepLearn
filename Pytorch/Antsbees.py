#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：Antsbees
   Description :  使用预训练模型 resnet 分类蚂蚁和蜜蜂  
    1. 数据集 https://download.pytorch.org/tutorial/hymenoptera_data.zip
    2. https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
    3. http://pytorch.org/docs/master/torchvision/transforms.html
    4. https://towardsdatascience.com/transfer-learning-using-pytorch-4c3475f4495
    5. https://towardsdatascience.com/transfer-learning-using-pytorch-part-2-9c5b18e15551
    6. http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
   Email : autuanliu@163.com
   Date：2018/3/22
"""
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Module, functional as F
from torchvision import transforms
from torchvision.models import resnet18
from functools import wraps

# Decorator test
def info(func):
    @wraps(func)
    def wrapper():
        print('GPU is available!')
        return func()
    return wrapper

# GPU
@info
def use_gpu():
    return torch.cuda.is_available()

# Data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Transfer Learning
model_conv = resnet18(pretrained=True)
# 固定参数, 使其不可更新或学习, 这使得只能更新最后一层的参数
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
# 获取 fc 层的输入特征数
num_ftrs = model_conv.fc.in_features
# 构建(修改)新的全连接层 fc (对原先存在的进行更改), 使其输出特征个数为 2
model_conv.fc = nn.Linear(num_ftrs, 2)

if use_gpu():
    model_conv = model_conv.cuda()   

# Training Model
if use_gpu():
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
else:
    inputs, labels = Variable(inputs), Variable(labels)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# zero the parameter gradients
optimizer.zero_grad()

# forward 
outputs = model(inputs)
loss = criterion(outputs, labels)

# backward + optimize only if in training phase
if phase == 'train':
    loss.backward()
    optimizer.step()

# Decaying Learning Rate
def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
