import torch
from torch import nn
from torch.nn import  functional as F

x = torch.arange(4)
print(x)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'x-file')
x2, y2 = torch.load('x-file')
print("x2==>%s \n" % x2, "y2==>%s" % y2)

"""
    可以写入或读取从字符串映射到张量的字典。 当我们要读取或写入模型中的所有权重时，这很方便。
"""
print("========================================================================")
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)


print("=========================================================================")
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')


clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

print(Y == clone(X))