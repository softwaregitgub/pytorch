"""
    通过深度学习框架的高级API也能更方便地实现softmax回归模型。
"""

import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""
    1、初始化模型参数
        Softmax 回归的输出层是一个全连接层
"""

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights);

"""
    2、损失函数
        在交叉熵损失函数中传递未归一化的预测，并同时计算softmax及其对数
"""

loss = nn.CrossEntropyLoss(reduction='none')

"""
    3、优化算法
        使用学习率为0.1的小批量随机梯度下降作为优化算法
"""

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

"""
    4、训练
        调用 之前 定义的训练函数来训练模型
"""
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)