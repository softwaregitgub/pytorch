import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

X = torch.rand(2, 20)
print(net(X).shape)

"""
    2.
        顺序块
"""


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X


net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print(net(X).shape)