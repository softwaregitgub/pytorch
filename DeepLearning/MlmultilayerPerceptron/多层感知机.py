"""
    一、激活函数
        激活函数（activation function）通过计算加权和并加上偏置来确定神经元是否应该被激活， 它们将输入信号转换为输出的可微运算。
        大多数激活函数都是非线性的。
"""

import torch
from d2l import torch as d2l

"""
    1、ReLU函数
        最受欢迎的激活函数是修正线性单元（Rectified linear unit，ReLU），
        1)它实现简单
        2)同时在各种预测任务中表现良好。
        3)ReLU提供了⼀种⾮常简单的⾮线性变换。给定元素x，ReLU函数被定义为该元素与0的最⼤值：
                    ReLU(x) = max(x, 0)
                    
    2、使⽤ReLU的原因是:
            1)它求导表现得特别好：要么让参数消失，要么让参数通过, 这使得优化表现得更好
            2)并且ReLU减轻了困扰以往神经⽹络的梯度消失问题
"""


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

"""
    1.2
        绘制ReLU函数的导数
"""
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

"""
    2、
        sigmoid函数
           对于一个定义域在 R 中的输入， sigmoid函数将输入变换为区间(0, 1)上的输出。 
           因此，sigmoid通常称为挤压函数（squashing function）： 
                它将范围（-inf, inf）中的任意输入压缩到区间（0, 1）中的某个值：
"""

y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

"""
    2.1
        sigmoid函数的导数图像如下所示。 
            注意，当输入为0时，sigmoid函数的导数达到最大值0.25； 
            而输入在任一方向上越远离0点时，导数越接近0。
"""

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

"""
    3、
        tanh函数
            与sigmoid函数类似， tanh(双曲正切)函数也能将其输入压缩转换到区间(-1, 1)上。
"""

y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

"""
    3.1
        tanh函数的导数
"""

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

