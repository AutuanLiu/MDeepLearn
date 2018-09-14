"""
Email: autuanliu@163.com
Date: 2018/9/14
"""

import torch

x = torch.randn(4, 3)
print(x.numel())
y = torch.full((2, 3), 3.141592)
print(y)
z = torch.cat((x, x), 1)
print(z)
t1 = torch.chunk(z, 2, 1)
print(f't1 is equal:\n{t1}')
indices = torch.tensor([0, 2])
q = torch.index_select(x, 0, indices)
print(q)
mask = x.ge(0.5)
e = torch.masked_select(x, mask)
print(e)

x1 = torch.tensor([0, 1, 2, 0, 3, 0])
x2 = torch.nonzero(x1)
print(x1, x2)
src = torch.tensor([[4, 3, 5], [6, 7, 8]])
y1 = torch.take(src, torch.tensor([0, 2, 5]))
print(y1)
src1 = torch.unbind(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print(src1)
x = torch.tensor([1, 2, 3, 4])
print(torch.unsqueeze(x, 0))