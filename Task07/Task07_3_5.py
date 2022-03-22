import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

transform_train = transforms.Compose(
    [transforms.ToTensor()])
transform_test = transforms.Compose(
    [transforms.ToTensor()])

train_data = datasets.CIFAR10(".", train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(".", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

images, labels = next(iter(train_loader))

# # 仅查看一张图片
# writer = SummaryWriter('./pytorch_tb')
# writer.add_image('images[0]', images[0])
# writer.close()
# Task07/2.png

# 将多张图片拼接成一张图片，中间用黑色网格分割
# # create grid of images
# writer = SummaryWriter('./pytorch_tb')
# img_grid = torchvision.utils.make_grid(images)
# writer.add_image('image_grid', img_grid)
# writer.close()
#
# Task07/3.png


# # 将多张图片直接写入
# writer = SummaryWriter('./pytorch_tb')
# writer.add_images("images",images,global_step = 0)
# writer.close()

# (tf) tj@tj-N8xEJEK:~/Study/pytorch$ tensorboard --logdir=./pytorch_tb
# TensorFlow installation not found - running with reduced feature set.
# TensorBoard 1.15.0 at http://tj-N8xEJEK:6006/ (Press CTRL+C to quit)


# writer = SummaryWriter('./pytorch_tb')
# for i in range(500):
#     x = i
#     y = x**2
#     writer.add_scalar("x", x, i) #日志中记录x在第step i 的值
#     writer.add_scalar("y", y, i) #日志中记录y在第step i 的值
# writer.close()
# Task07/4.png



# 如果想在同一张图中显示多个曲线，则需要分别建立存放子路径（使用SummaryWriter指定路径即可自
# 动创建，但需要在tensorboard运行目录下），同时在add_scalar中修改曲线的标签使其一致即可：
# writer1 = SummaryWriter('./pytorch_tb/x')
# writer2 = SummaryWriter('./pytorch_tb/y')
# for i in range(500):
#     x = i
#     y = x*2
#     writer1.add_scalar("same", x, i) #日志中记录x在第step i 的值
#     writer2.add_scalar("same", y, i) #日志中记录y在第step i 的值
# writer1.close()
# writer2.close()
# Task07/5.png


import torch
import numpy as np

def norm(mean, std):
    t = std * torch.randn((100, 20)) + mean
    return t
writer = SummaryWriter('./pytorch_tb')

for step, mean in enumerate(range(-10, 10, 1)):
    w = norm(mean, 1)
    writer.add_histogram("w", w, step)
    writer.flush()
writer.close()
# Task07/6.png