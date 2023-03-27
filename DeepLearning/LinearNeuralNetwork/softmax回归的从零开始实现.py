

import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


"""
   1、初始化模型参数
    1）展平每个图像，把它们看作长度为784的向量；
    2）因为我们的数据集有10个类别，所以网络输出维度为10。 
    3）权重将构成一个 784*10 的矩阵
    4）偏置将构成一个 1*10 的行向量
"""

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 使用正态分布初始化我们的权重W
b = torch.zeros(num_outputs, requires_grad=True)  # 偏置初始化为0

"""
    2、定义softmax操作
"""
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

"""
    输出:
        tensor([[5., 7., 9.]]) 
        tensor([[ 6.],
                [15.]])
"""

"""
    实现softmax由三个步骤组成：
        1)对每个项求幂（使用exp）；
        2)对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
        3)将每一行除以其规范化常数，确保结果的和为1。

"""


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


X = torch.normal(0, 1, (2, 5))
print(X)
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))

"""
    注意:
        虽然这在数学上看起来是正确的，但我们在代码实现中有点草率。 
        矩阵中的非常大或非常小的元素可能造成数值上溢或下溢，但我们没有采取措施来防止这点。
        解决办法：插入一行：
            X -= X.max()
"""

"""
    3、定义模型
        1)下面的代码定义了输入如何通过网络映射到输出。 
        2)注意:
            将数据传递到模型之前，我们使用reshape()函数将每张原始图像展平为向量
"""

print("*************************************")
# print(W, W.shape)
# print(W.shape[0])
# print(torch.matmul(X.reshape((-1, W.shape[0])), W))


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


"""
    4、定义损失函数
"""

y = torch.tensor([0, 2])
# print(y)
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[0, 1])
# print(y_hat[[0, 1], y])
# print(len(y_hat))
# print(y_hat[range(len(y_hat))])
"""
    实现交叉熵损失函数
"""


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


cross_entropy(y_hat, y)

"""
    5、分类精度
    5.1
        argmax返回的是最大数的索引。
        argmax有一个参数axis，默认是0，表示每一列的最大值的索引，axis=1表示每一行的最大值的索引。
"""


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)    # 用argmax获得每行中最大元素的索引来获得预测类别
    cmp = y_hat.type(y.dtype) == y      # 将y_hat的数据类型转换为与y的数据类型一致
    return float(cmp.type(y.dtype).sum())


print(accuracy(y_hat, y) / len(y))



"""
    5.2
        评估在任意模型 net 的准确率
"""


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    # print("net:",net)
    with torch.no_grad():
        for X, y in data_iter:
            # print("accuracy(net(X), y)-->", accuracy(net(X), y), "y.numel()-->", y.numel())
            metric.add(accuracy(net(X), y), y.numel())
    # print(metric[1])
    return metric[0] / metric[1]


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


print(evaluate_accuracy(net, test_iter))


"""
    6、训练
"""


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


"""
    6.2
    定义一个在动画中绘制数据的实用程序类
"""


class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


"""
    6.3 
    实现一个训练函数
        它会在train_iter访问到的训练数据集上训练一个模型net。 
        该训练函数将会运行多个迭代周期（由num_epochs指定）。 
        在每个迭代周期结束时，利用test_iter访问到的测试数据集对模型进行评估。 
        我们将利用Animator类来可视化训练进度。
"""


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


"""
    7、预测
    
"""
def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)