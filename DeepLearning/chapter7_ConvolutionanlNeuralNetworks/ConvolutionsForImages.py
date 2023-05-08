import torch
from torch import nn

"""
    2.1
        互相关运算
"""


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))

"""
    2.2
        卷积层
"""


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
#print(Y)

#print(corr2d(X.t(), K))

"""
    2.4 学习卷积核
"""
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = X.reshape(1, 1, 6, 8)
print(X)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y - Y_hat) ** 2
    conv2d.zero_grad()
    l.sum().backward()

    conv2d.weight.data[:] -= lr * conv2d.weight.grad

    if(i+1) % 2 == 0:
        print(f'epoch {i+1}, loss{l.sum():.3f}')

print(conv2d.weight.data[:])

""""
    import torch
    a = torch.randn(size=(), requires_grad=True)
    b = torch.randn(size=(), requires_grad=True)
    c = torch.randn(size=(), requires_grad=True)
    
    c = a * b
    c.backward()
    
    
    print( a.grad == b,a)
    print( b.grad == a,b)
    
    
    output:
    tensor(True) tensor(-0.7874, requires_grad=True)
    tensor(True) tensor(0.0025, requires_grad=True)
    
    若在 torch 中 对定义的变量 requires_grad 的属性赋为 True ，那么此变量即可进行梯度以及导数的求解，在以上代码中，a,b,c 都可以理解为数学中的x,y,z进行运算，c 分别对 a,b 求导的结果为 b,a。

    当c.backward() 语句执行后，会自动对 c 表达式中 的可求导变量进行方向导数的求解，并将对每个变量的导数表达存储到 变量名.grad 中。
    
    可如此理解 c.backward() = a.grad I + b.grad j

"""


