"""
Email: autuanliu@163.com
Date: 2018/9/13
"""

import torch
from torch.autograd import Function
from torch.nn import Module

# 创建一个 tensor, requires_grad 默认为 False(浮点数)
x = torch.randn(5, 5)
print(x, x.requires_grad)
y = torch.randn((5, 5), requires_grad=True)
print(y, y.requires_grad)
z = x + y
z1 = x + x
print(z.requires_grad)
print(z1.requires_grad)

# 在每次迭代中，图形都是从头重新创建的，这正是允许使用任意的Python控制流语句的原因，它可以在每次迭代中改变图形的整体形状和大小
# 在你开始训练之前，你不需要对所有可能的路径进行编码

# 广播原则
# 1. 每个张量至少有一个维度
# 2. 当在维度大小上迭代时，从后面的维度开始，维度大小必须是相等的，或者其中一个是 1，或者其中一个不存在。
# 3. 相同大小的张量一定是可以广播的
x1 = torch.randn(5, 3, 4, 1)
y1 = torch.randn(   3, 1, 1)
print(x1, y1, x1 + y1)
# x and y are broadcastable.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist

# but:
x = torch.empty(5,2,4,1)
y = torch.empty(  3,1,1)
# x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3
# 广播计算后使用最大的维度作为最终的结果
print(x1.add_(y1))

y2 = torch.add(torch.ones(4,1), torch.randn(4))
print(y2)
y3 = torch.add(torch.ones(4,1), torch.randn(4, 1))
print(y3)

# 一旦一个张量被分配，你就可以对它进行操作，不管选择的设备是什么，结果总是会被放置在与 tensor 相同的设备上
# 默认情况下，cross-GPU是不被支持的
# 除非你启用点对点的内存访问，否则任何试图在不同设备上运行的张量上的操作都会引发一个错误
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(dev)
x3 = torch.tensor([3, 3, 2.3], device=dev)
print(x3, x3.type())
y3 = torch.tensor(5)
y4 = y3.to(dev)
print(y3, y4, y3.cuda(), y4.cpu())
y_cpu = x.new_full([3, 2], fill_value=0.3)
print(y_cpu)
y_cpu = x.cuda().new_full([3, 2], fill_value=0.3)
print(y_cpu)
y_cpu = torch.ones_like(x)
print(y_cpu)

# Extending torch.autograd
# Inherit from Function
class LinearFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias) # 先执行这一句
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

# Module 的扩充需要实现 __init__() 和 forward()
