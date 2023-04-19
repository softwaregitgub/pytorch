import torch
from torch import nn

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
print(torch.cuda.device_count())


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu())
print(try_gpu(10))
print(try_all_gpus())

"""
    2、
        张量与GPU
            我们可以查询张量所在的设备。 默认情况下，张量是在CPU上创建的
"""
x = torch.tensor([1, 2, 3])
print(x.device)

"""
    2.1 
        存储在GPU上
"""
X = torch.ones(2, 3, device=try_gpu())
print(X)

Y = torch.rand(2, 3, device='cuda:0')
print(Y)

print(X+Y)

"""
    3、
        神经网络与GPU
"""
print("===========================================")
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)
"""
    >>> net[0].weight.data.device
    device(type='cuda', index=0)
"""

