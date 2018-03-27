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
    7. https://github.com/svishnu88/pytorch
    8. https://chsasank.github.io/
    Email : autuanliu@163.com
    Date：2018/3/22
"""
import copy
import time
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet18

# interactive mode
plt.ion()
# GPU
use_gpu = torch.cuda.is_available()
torch.manual_seed(5)

# Data augmentation and normalization
data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

# Load Data
data_dir = PurePath('../datasets/antsbees')
# 字典推导式
image_datasets = {x: datasets.ImageFolder(data_dir / x, data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


def get_cuda_data(input_d, label_d):
    if use_gpu:
        print('GPU is available!')
        inputs, labels = Variable(input_d.cuda()), Variable(label_d.cuda())
    else:
        inputs, labels = Variable(input_d), Variable(label_d)
    return inputs, labels


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # pause a bit so that plots are updated
    plt.pause(0.001)


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# 训练模型
def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    """训练模型
    
    Parameters:
    ----------
    model : nn.Module object
    criterion : {nn.Module object
        损失函数
    optimizer : optim object
        优化器
    scheduler : optim.lr_scheduler object
        学习率调度器
    num_epochs : int, optional
        epoch 数量 (the default is 25)
    
    Returns
    -------
    nn.Module object
        返回训练好的模型
    """
    start = time.time()
    # 深度复制模型的权重值
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}\n{'*' * 50}")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # 根据调度器更新学习率
                scheduler.step()
                # Set model to training mode
                model.train(True)
            else:
                # Set model to evaluate mode
                model.train(False)
            # 每个 epoch 都要记录其损失和预测正确的个数
            running_loss, running_corrects = 0.0, 0

            # Iterate over data. (minibatch)
            for inputs, labels in dataloaders[phase]:
                # wrap torch data in Variable
                inputs, labels = get_cuda_data(inputs, labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                # 预测的类别
                _, preds = torch.max(outputs.data, 1)
                # 损失的前向传播
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 每个 minibatch 统计分析结果(总损失)
                running_loss += loss.data[0] * inputs.size()[0]
                running_corrects += torch.sum(preds == labels.data)
            # 每个 epoch 计算 train and valid 的 loss and acc
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val Acc: {best_acc}')

    # load and return the best model weights
    model.load_state_dict(best_model_wts)
    return model


# Visualizing the model predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, (inputs, labels) in enumerate(dataloaders['val']):
        inputs, labels = get_cuda_data(inputs, labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title(f'predicted: {class_names[preds[j]]}')
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)


################################################################
# 训练并更新全部的参数
# 预训练模型
# model_ft = resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)

# if use_gpu:
#     print('GPU is available!')
#     model_ft = model_ft.cuda()

# criterion = nn.CrossEntropyLoss()

# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
# 仅保存和加载模型参数(推荐使用)
# torch.save(model_ft.state_dict(), 'all_params.pkl')
# visualize_model(model_ft)
###############################################################

###############################################################
# Transfer Learning
model_conv = resnet18(pretrained=True)
# 固定参数, 使其不可更新或学习, 这使得只能更新最后一层的参数(一定要在更改模型之前)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
# 获取 fc 层的输入特征数
num_ftrs = model_conv.fc.in_features
# 构建(修改)新的全连接层 fc (对原先存在的进行更改), 使其输出特征个数为 2
model_conv.fc = nn.Linear(num_ftrs, 2)

# Training Model
if use_gpu:
    print('GPU is available!')
    model_conv = model_conv.cuda()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=8, gamma=0.1)

# Train and evaluate
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=1)
visualize_model(model_conv)
# 仅保存和加载模型参数(推荐使用)
torch.save(model_conv.state_dict(), 'trans_params.pkl')
plt.ioff()
plt.show()
