import torch

size=10000
a=torch.rand(size, size)
b=torch.rand(size, size)
c=torch.mm(a, b)
