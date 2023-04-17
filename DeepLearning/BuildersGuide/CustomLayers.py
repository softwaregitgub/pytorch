import torch
import torch.nn.functional as F
from torch import nn

"""
    5.4.1
        不带参数的层
"""


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


"""让我们向该层提供一些数据，验证它是否能按预期工作。"""
layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))


"""现在，我们可以将层作为组件合并到更复杂的模型中。"""
net = nn.Sequential(nn.Linear(8, 128),
                    CenteredLayer())

Y = net(torch.rand(size=(4, 8)))
print(Y.mean())
print(Y.size())


"""
    5.4.2. 
        带参数的层
"""


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bais = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bais
        return F.relu(linear)


linear = MyLinear(5, 3)
print(linear.weight)

print(linear(torch.randn(2, 5)))

"""使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层"""
print("==================================================")
net = nn.Sequential(MyLinear(64, 8),
                    MyLinear(8, 1))

print(net(torch.rand(2, 64)))