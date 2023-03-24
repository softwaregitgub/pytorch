"""
    1、生成数据集
    通过使用深度学习框架来简洁地实现 线性回归模型 生成数据集
"""
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b , 1000)

"""
    2、读取数据
    调用框架中现有的API
    1)next()：返回迭代器的下一个项目。next() 函数要和生成迭代器的 iter() 函数一起使用。
    2)next(iterable[, default])：
        iterable -- 可迭代对象；
        default -- 可选，用于设置在没有下一个元素时返回该默认值，如果不设置，又没有下一个元素则会触发 StopIteration 异常。
"""


def load_array(date_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*date_arrays)                      # 星号表示解开list写入其中参数。
    return data.DataLoader(dataset, batch_size, shuffle=is_train)   # shuffle：是否需要随机打乱数据。


batch_size = 10
data_iter = load_array((features, labels), batch_size)              # 第一个参数位置传进去了一个list，和后面的next有关

print(next(iter(data_iter)))

"""
[tensor([[ 0.7164,  0.4257],
        [ 0.1860, -1.3525],
        [ 0.1581, -1.1173],
        [-0.2142, -0.2856],
        [-0.5216, -1.9521],
        [-0.6183,  0.4979],
        [ 1.1302, -0.1235],
        [ 1.3563,  1.8829],
        [-1.9732,  0.0161],
        [ 1.1374, -0.9130]]), tensor([[4.1846],
        [9.1800],
        [8.3127],
        [4.7548],
        [9.7948],
        [1.2612],
        [6.8958],
        [0.5016],
        [0.2050],
        [9.5805]])]
epoch 1, loss 0.000165
epoch 2, loss 0.000095
epoch 3, loss 0.000095
w的估计误差： tensor([-4.4966e-04,  3.1233e-05])
b的估计误差： tensor([9.5844e-05])


"""

"""
    3、定规模型
        使用框架的预定义好的层
        1)在PyTorch中，全连接层在Linear类中定义。
        2)第一个指定输入特征形状，即2.
        3)第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
"""


from torch import nn
net = nn.Sequential(nn.Linear(2, 1))  # 把后面的层放到一个list of layers里。

"""
    4、初始化模型参数
        1)在使用net之前，我们需要初始化模型参数。 
        2)这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 偏置参数将初始化为零。
        3)通过net[0]选择网络中的第一个图层， 然后使用weight.data和bias.data方法访问参数。 
        我们还可以使用替换方法normal_和fill_来重写参数值。
        4)等价于之前的以下代码：
            w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
            b = torch.zeros(1, requires_grad=True)
"""
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
"""
    结果：
        net[0].bias.data.fill_(0) --> tensor([0.])
"""

"""
    5、定义损失函数
        计算均方误差使用的是MSELoss类，也称为平方范数。 默认情况下，它返回所有样本损失的平均值。
"""
loss = nn.MSELoss()

"""
    6、定义优化算法
        1）小批量随机梯度下降算法是一种优化神经网络的标准工具
        2）当我们实例化一个SGD实例时，我们要指定优化的参数（可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。 
        3）小批量随机梯度下降只需要设置lr值，这里设置为0.03。
"""
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

"""
    7、训练
        每个迭代周期里，我们将完整遍历一次数据集（train_data）， 不停地从中获取一个小批量的输入和相应的标签。 
        对于每一个小批量，我们会进行以下步骤:
            1)通过调用net(X)生成预测并计算损失l（前向传播）。
            2)通过进行反向传播来计算梯度。
            3)通过调用优化器来更新模型参数。

"""

num_epoch = 3
for epoch in range(num_epoch):
        for X, y in data_iter:
            l = loss(net(X), y)  # net本身带了参数w和b，所以不用把w和b传进去了。
            trainer.zero_grad()  # 优化器梯度清零。
            l.backward()
            trainer.step()       # 进行一次模型更新。
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

"""
    迭代过程：
        epoch 1, loss 0.000165
        epoch 2, loss 0.000095
        epoch 3, loss 0.000095
"""

"""
    8、结果评价
    比较生成数据集的真实参数和通过有限数据训练获得的模型参数
"""

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

"""
    结果：
        w的估计误差： tensor([-4.4966e-04,  3.1233e-05])
        b的估计误差： tensor([9.5844e-05])
"""