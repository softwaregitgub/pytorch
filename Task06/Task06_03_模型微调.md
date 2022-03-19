# pytorch
pytorch study
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
opt = parser.parse_args()


if opt.adam:
optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
else:
optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
print("Optimizer:")
print(optimizer)



# 选择一种优化器
optimizer = torch.optim.Adam(...)
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler....
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# 进行训练
for epoch in range(100):
train(...)
validate(...)
optimizer.step()
# 需要在优化器参数更新之后再动态调整学习率
scheduler1.step()
...
schedulern.step()
#注：我们在使用官方给出的torch.optim.lr_scheduler时，需要将scheduler.step()放在optimizer.step()后面进行使用。



#
def adjust_learning_rate(optimizer, epoch):
lr = args.lr * (0.1 ** (epoch // 30))
for param_group in optimizer.param_groups:
param_group['lr'] = lr
#有了adjust_learning_rate函数的定义，在训练的过程就可以调用我们的函数来实现学习率的动态变化
def adjust_learning_rate(optimizer,...):
...
optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9)
for epoch in range(10):
train(...)
validate(...)
adjust_learning_rate(optimizer,epoch)
