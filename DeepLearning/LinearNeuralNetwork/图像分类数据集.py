"""
    MNIST数据集是图像分类中广泛使用的数据集之一，但作为基准数据集过于简单。
    我们将使用类似但更复杂的Fashion-MNIST数据集
"""

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


d2l.use_svg_display()

"""
    1.1、读取数据集
        通过框架中的内置函数将 Fashion-MNIST 数据集下载并读取到内存中。
        1）通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
           并除以255使得所有像素的数值均在0～1之间。
        2）注：../data是上一级目录的data文件夹，./data是当前目录（即当前正在运行的代码文件的目录）的data文件夹。
"""

trans = transforms.ToTensor()   # 把图片转为tensor。

mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)

mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
"""
    Python 3.9.16 (main, Mar  8 2023, 14:00:05) 
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ../data/FashionMNIST/raw/train-images-idx3-ubyte.gz
    100.0%
    Extracting ../data/FashionMNIST/raw/train-images-idx3-ubyte.gz to ../data/FashionMNIST/raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
    100.0%
    Extracting ../data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ../data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
    100.0%
    Extracting ../data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ../data/FashionMNIST/raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
    100.0%
    Extracting ../data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw

"""

"""
    Fashion-MNIST由10个类别的图像组成， 
    每个类别由训练数据集（train dataset）中的6000张图像 和测试数据集（test dataset）中的1000张图像组成。 
    因此，训练集和测试集分别包含60000和10000张图像。 
    测试数据集不会用于训练，只用于评估模型性能。
"""
print(len(mnist_train), len(mnist_test))        # 60000 10000

"""
    每个输入图像的高度和宽度均为28像素。 数据集由灰度图像组成，其通道数为1。
    
"""
print(mnist_train[0][0].shape)  # torch.Size([1, 28, 28])


"""
    1.2、两个可视化数据集的函数
        Fashion-MNIST中包含的10个类别，分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、
        dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、
        bag（包）和ankle boot（短靴）。 以下函数用于在数字标签索引及其文本名称之间进行转换。
"""


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


"""
    subplot是MATLAB中的函数，是将多个图画到一个平面上的工具
    plt.subplot()函数用于直接指定划分方式和位置进行绘图。返回一个包含figure和axes对象的元组。
    MATLAB和pyplot有当前的图形（figure）和当前的轴（axes）的概念，所有的作图命令都是对当前的对象作用。
    可以通过gca()获得当前的轴（axes），通过gcf()获得当前的图形（igure）。
"""


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)   # figure , axex , 把父图分成num_rows*num_cols个子图
    axes = axes.flatten()           # 把子图展开赋值给axes,axes[0]便是第一个子图，axes[1]是第二个
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


"""
    1.3、几个样本的图像及其相应的标签    
"""

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))

show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));


"""
    2、读取小批量
        大小为batch_size
"""

batch_size = 256


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

"""
    2.1 读取训练数据所需的时间
        1.73 sec
"""
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')


"""
    3、整合所有组件
        定义load_data_fashion_mnist函数，用于获取和读取Fashion-MNIST数据集。 
        这个函数返回训练集和验证集的数据迭代器。 
        此外，这个函数还接受一个可选参数resize，用来将图像大小调整为另一种形状。
"""


def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


"""
    3.1
        我们通过指定resize参数来测试load_data_fashion_mnist函数的图像大小调整功能。
"""

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

"""
    torch.Size([32, 1, 64, 64]) torch.float32 torch.Size([32]) torch.int64
"""