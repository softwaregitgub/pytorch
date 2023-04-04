import torch
from torch import nn
from d2l import torch as d2l

"""
    1.0
        从零开始实现
"""

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

"""
        >>> torch.rand(X.shape)
        tensor([[0.6411, 0.0154, 0.0679, 0.0679, 0.6266, 0.5316, 0.1535, 0.7098],
                [0.8631, 0.7957, 0.4254, 0.0121, 0.5649, 0.7364, 0.6978, 0.0509]])
        >>> X
        tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
                [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
        >>> dropout = 0.5
        >>> mask = (torch.rand(X.shape) > dropout).float()
        >>> mask
        tensor([[0., 1., 1., 1., 0., 1., 1., 1.],
                [0., 0., 0., 0., 1., 1., 0., 0.]])
        >>> mask = (torch.rand(X.shape) > dropout)
        >>> mask
        tensor([[ True, False, False,  True,  True,  True,  True, False],
                [ True, False,  True,  True,  True, False,  True, False]])

"""

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))

"""
    tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
    tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
    tensor([[ 0.,  0.,  0.,  6.,  0.,  0.,  0., 14.],
            [16., 18., 20., 22.,  0.,  0., 28.,  0.]])
    tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.]])
"""

"""
    1.1
        定义模型参数
"""

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

"""
    1.2
        定义模型
"""

dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


"""
    1.2
        定义模型
"""
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

"""
    1.3、
        训练和测试
"""
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)