import torch
from torch import nn
from d2l import torch as d2l

"""
    1、模型
        第一层：隐藏层 包含256个隐藏单元，并使用了ReLU激活函数。
        第二层：输出层
        
        torch.nn.Flatten():
            因为其被用在神经网络中，输入为一批数据，第一维为batch,
            通常要把一个数据拉成一维，而不是将一批数据拉为一维。
            所以torch.nn.Flatten()默认从第二维开始平坦化。
"""

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights);

"""
    2、训练过程
"""

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)