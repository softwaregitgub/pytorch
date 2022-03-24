
# 2 PyTorchVideo（视频）
#
#     简介：PyTorchVideo是一个专注于视频理解工作的深度学习库，提供加速视频理解研究所需的可重用、模块化和高效的组件，
#     使用PyTorch开发，支持不同的深度学习视频组件，如视频模型、视频数据集和视频特定转换。
#
#     特点：基于PyTorch，提供Model Zoo，支持数据预处理和常见数据，采用模块化设计，支持多模态，优化移动端部署
#
#     使用方式：TochHub、PySlowFast、PyTorch Lightning

#我们知道在计算机视觉中处理的数据集有很大一部分是图片类型的，
#如果获取的数据是格式或者大小不一的图片，
#则需要进行归一化和大小缩放等操作，这些是常用的数据预处理方法。
from torchvision import transforms
data_transform = transforms.Compose([
    transforms.ToPILImage(),   # 这一步取决于后续的数据读取方式，如果使用内置数据集则不需要
    transforms.Resize(image_size),
    transforms.ToTensor()
])