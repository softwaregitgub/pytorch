import torch
from torch import nn

net = nn.Sequential(
                    nn.Linear(4, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1))

X = torch.rand(size=(2, 4))
print(net(X))

"""参数访问"""
print(net[2].state_dict())


"""目标参数"""
print("======================================================")
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print("============================================================")
print(net[2].weight.grad == None)

"""一次性访问所有参数"""
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print("net.state_dict()['2.bias'].data ==>", net.state_dict()['2.bias'].data)


"""
    3、
        从嵌套块收集参数
"""
print("============================================================")

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential();
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)

print("访问第一个主要的块中、第二个子块的第一层的偏置项==>", '\n', rgnet[0][1][0].weight.data)

"""
    2、参数初始化
        想要针对里面的某个特定的模块调用函数可以加入type(m) == nn.Linear:这类判断语句
"""


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean= 0, std= 0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print("net[0]  ==> ", net[0].weight.data[0], '\n', net[0].bias.data[0])


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print("constant net[0]==> ", net[0].weight.data[0], '\n', net[0].bias.data[0])

print("==========================================================")


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(init_xavier)
net[2].apply(init_42)

print(net[0].weight.data[0])
print(net[2].weight.data)


"""
    5.2 自定义初始化
"""
print("================================================================================")


def my_init(m):
    if type(m) == nn.Linear:
        print("init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(net[0].weight[:3])


"""
    2.3 参数绑定
        这个例子表明第三个和第五个神经网络层的参数是绑定的。 
        它们不仅值相等，而且由相同的张量表示。 因此，如果我们改变其中一个参数，另一个参数也会改变。 
        这里有一个问题：当参数绑定时，梯度会发生什么情况？ 
        答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层 （即第三个神经网络层）
        和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。
"""
print("=================================================================================")
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))

net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
print(net[6])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
print(net)