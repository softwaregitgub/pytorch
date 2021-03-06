
# 3 模型微调
#
#  概念：找到一个同类已训练好的模型，调整模型参数，使用数据进行训练。
#
# 模型微调的流程
#         1、在源数据集上预训练一个神经网络模型，即源模型
#         2、创建一个新的神经网络模型，即目标模型，该模型复制了源模型上除输出层外的所有模型设计和参数
#         3、给目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化改成的模型参数
#         4、使用目标数据集训练目标模型


import torchvision.models as models



#    传递pretrained参数
#
# 通过True或者False来决定是否使用预训练好的权重，在默认状态下pretrained = False，意味着我们不使用预训练得到的权重，
# 当pretrained = True，意味着我们将使用在一些数据集上预训练得到的权重。

#6.3.3 训练特定层

import torchvision.models as models
import torch.nn as nn
import torch

def set_parameter_requires_grad(model, feature_extracting):
if feature_extracting:
for param in model.parameters():
param.requires_grad = False

# 冻结参数的梯度
feature_extract = True
model = models.resnet18(pretrained=True)
set_parameter_requires_grad(model, feature_extract)
# 修改模型
num_ftrs = model.fc.in_features
model.fc = nn.Linear(in_features=512, out_features=4, bias=True)

#可以看到最后一层修改成我们所需要的4分类，而不是原先的1000分类了
print(model)

##################################################################################

/home/tj/anaconda3/envs/tf/bin/python /home/tj/Study/pytorch/Task06_03.py
##Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /home/tj/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
#100.0%
#ResNet(
#		(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#		(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(relu): ReLU(inplace=True)
#		(maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#		(layer1): Sequential(
#		(0): BasicBlock(
#		(conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(relu): ReLU(inplace=True)
#		(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		)
#		(1): BasicBlock(
#		(conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(relu): ReLU(inplace=True)
#		(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		)
#		)
#		(layer2): Sequential(
#		(0): BasicBlock(
#		(conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#		(bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(relu): ReLU(inplace=True)
#		(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(downsample): Sequential(
#		(0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#		(1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		)
#		)
#		(1): BasicBlock(
#		(conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(relu): ReLU(inplace=True)
#		(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		)
#		)
#		(layer3): Sequential(
#		(0): BasicBlock(
#		(conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#		(bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(relu): ReLU(inplace=True)
#		(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(downsample): Sequential(
#		(0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#		(1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		)
#		)
#		(1): BasicBlock(
#		(conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(relu): ReLU(inplace=True)
#		(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		)
#		)
#		(layer4): Sequential(
#		(0): BasicBlock(
#		(conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#		(bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(relu): ReLU(inplace=True)
#		(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(downsample): Sequential(
#		(0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#		(1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		)
#		)
#		(1): BasicBlock(
#		(conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		(relu): ReLU(inplace=True)
#		(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#		(bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#		)
#		)
#		(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#		(fc): Linear(in_features=512, out_features=4, bias=True)
#		)
进程已结束,退出代码0

