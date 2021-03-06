# 6.4 半精度训练

# 半精度优势：减少显存占用，提高GPU同时加载的数据量
#
# 设置半精度训练：
# 导入torch.cuda.amp的autocast包
# 在模型定义中的forward函数上，设置autocast装饰器
# 在训练过程中，在数据输入模型之后，添加with
# autocast()
#
# 适用范围：适用于数据的size较大的数据集（比如3D图像、视频等）

#感觉一般情况下也是用不到的，主要还是因为计算机性能不够，无法一次性加载
from torch.cuda.amp import autocast

@autocast()
def forward(self, x):
...
return x

for x in train_loader:
x = x.cuda()
with autocast():
output = model(x)
...
