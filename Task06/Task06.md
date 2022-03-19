import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 使用GPU环境
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 配置其他超参数
batch_size = 256
# 在Windows环境下，需要将num_workers改为0，否则会存在多线程问题
num_workers = 0
lr = 1e-4
epochs = 20


from torchvision import transforms

# 设置数据变换
image_size = 28
data_transform = transforms.Compose([
transforms.ToPILImage(),   # 这一步取决于后续的数据读取方式，如果使用内置数据集则不需要
transforms.Resize(image_size),
transforms.ToTensor()
])

#自定义数据集


# csv数据下载链接：https://www.kaggle.com/zalando-research/fashionmnist
class FMDataset(Dataset):
def __init__(self, df, transform=None):
self.df = df
self.transform = transform
self.images = df.iloc[:, 1:].values.astype(np.uint8)
self.labels = df.iloc[:, 0].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image / 255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


train_df = pd.read_csv("./FashionMNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("./FashionMNIST/fashion-mnist_test.csv")

train_data = FMDataset(train_df, data_transform)
test_data = FMDataset(test_df, data_transform)

# 使用DataLoader类加载数据
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

import matplotlib.pyplot as plt
image, label = next(iter(train_loader))
print(image.shape, label.shape)
plt.imshow(image[0][0], cmap="gray")

torch.Size([256, 1, 28, 28])
torch.Size([256])


# 使用CNN
class Net(nn.Module):
def __init__(self):
super(Net, self).__init__()
self.conv = nn.Sequential(
nn.Conv2d(1, 32, 5),
nn.ReLU(),
nn.MaxPool2d(2, stride=2),
nn.Dropout(0.3),
nn.Conv2d(32, 64, 5),
nn.ReLU(),
nn.MaxPool2d(2, stride=2),
nn.Dropout(0.3)
)
self.fc = nn.Sequential(
nn.Linear(64 * 4 * 4, 512),
nn.ReLU(),
nn.Linear(512, 1)
)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x


model = Net()
model = model.cuda()


# 使用交叉熵损失函数
#criterion = nn.CrossEntropyLoss()


class DiceLoss(nn.Module):
def __init__(self, weight=None, size_average=True):
super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice




#criterion
criterion = DiceLoss()
# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
# 设置训练状态
model.train()
train_loss = 0
# 循环读取DataLoader中的全部数据
for data, label in train_loader:
# 将数据放到GPU用于后续计算
data, label = data.cuda(), label.cuda()
# 将优化器的梯度清0
optimizer.zero_grad()
# 将数据输入给模型
output = model(data)
# 设置损失函数
loss = criterion(output, label)
# 将loss反向传播给网络
loss.backward()
# 使用优化器更新模型参数
optimizer.step()
# 累加训练损失
train_loss += loss.item() * data.size(0)
train_loss = train_loss/len(train_loader.dataset)
print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

def val(epoch):
# 设置验证状态
model.eval()
val_loss = 0
gt_labels = []
pred_labels = []
# 不设置梯度
with torch.no_grad():
for data, label in test_loader:
data, label = data.cuda(), label.cuda()
output = model(data)
preds = torch.argmax(output, 1)
gt_labels.append(label.cpu().data.numpy())
pred_labels.append(preds.cpu().data.numpy())
loss = criterion(output, label)
val_loss += loss.item()*data.size(0)
# 计算验证集的平均损失
val_loss = val_loss/len(test_loader.dataset)
gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
# 计算准确率
acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))

for epoch in range(1, epochs+1):
train(epoch)
val(epoch)


#Training

/home/tj/anaconda3/envs/tf/bin/python /home/tj/Study/pytorch/Task04.py
torch.Size([256, 1, 28, 28]) torch.Size([256])
/home/tj/anaconda3/envs/tf/lib/python3.6/site-packages/torch/nn/functional.py:1625: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
Epoch: 1 	Training Loss: -0.628984
Epoch: 1 	Validation Loss: -0.635451, Accuracy: 0.100000

