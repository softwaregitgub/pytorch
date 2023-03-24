import random
import torch
from d2l import torch as d2l

"""
    1、生产数据集
    根据带有噪声的线性模型构造一个人造数据集。 
    y=Xw+b+ϵ
    
    torch.normal() --> 该函数返回从单独的正态分布中提取的随机数的张量，该正态分布的均值是mean，标准差是std。
    如： torch.normal(mean=0, std=1, size=(2, 2)) --> 我们从一个标准正态分布N～(0,1)，提取一个2x2的矩阵
    
    torch.matmul --> 是tensor的乘法，输入可以是高维的。
"""


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))      # >>> X.shape  torch.Size([1000, 2])
    y = torch.matmul(X, w) + b                          # >>> y.shape   torch.Size([1000])
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel:', labels[0])  # 注意，features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）。
print(features.shape)
"""
    输出结果：
            features: tensor([0.2206, 0.9954]) 
            label: tensor([1.2375])
            torch.Size([1000, 2])
"""

"""
    通过生成第二个特征features[:, 1]和labels的散点图， 可以直观观察到两者之间的线性关系。
"""
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1);


"""
    2、读取数据集
    1)定义一个data_iter 函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量. 
    每个小批量包含一组特征和标签。
    2)yield：可以看作等同return，看做return之后再把它看做一个是生成器（generator）的一部分（带yield的函数才是真正的迭代器），
    也即在循环中来回的返回值，一直到完成为止。
"""


def data_iter(batch_size, features, labels):
    num_examples = len(features)    # num_examples = 1000
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
"""
    输出结果：
                tensor([[ 0.0375, -0.7962],
                [-0.7670, -0.2316],
                [ 0.0117, -0.9196],
                [ 0.7010, -0.0037],
                [-1.7082, -0.4779],
                [-0.3089, -1.2139],
                [-0.6846, -1.0615],
                [ 1.7186,  1.8719],
                [-0.6272, -0.6038],
                [-0.1547, -0.0301]]) 
                tensor([[6.9870],
                [3.4456],
                [7.3278],
                [5.6082],
                [2.4085],
                [7.7217],
                [6.4347],
                [1.2853],
                [4.9863],
                [3.9962]])
"""

"""
    3、初始化模型参数
"""
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

"""
    4、定义模型
"""


def linreg(X, w, b):
    """线性回归模型。"""
    return torch.matmul(X, w) + b


"""
    5、定义损失函数
    1）因为需要计算损失函数的梯度，所以我们应该先定义损失函数。 
    2）在实现中，我们需要将真实值y的形状转换为和预测值y_hat的形状相同。
"""


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


"""
    6、定义优化算法
        该函数接受模型参数集合、学习速率和批量大小作为输入。每 一步更新的大小由学习速率lr决定。 
        因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（batch_size） 来规范化步长，
        这样步长大小就不会取决于我们对批量大小的选择。
"""


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


"""
    7、训练
    迭代周期个数num_epochs和学习率lr都是超参数
    
    概括一下，我们将执行以下循环：
        1）初始化参数
        2）重复以下训练，直到完成
            计算梯度
            更新参数
"""

lr = 0.03  # 学习率
num_epochs = 3  # 迭代周期个数（epoch）
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

"""
    8、结果
    我们可以通过比较真实参数和通过训练学到的参数来评估训练的成功程度。 事实上，真实参数和通过训练学到的参数确实非常接近。
"""

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')


"""
    迭代周期：
        epoch 1, loss 0.044128
        epoch 2, loss 0.000181
        epoch 3, loss 0.000051
    
    w的估计误差: tensor([ 0.0003, -0.0007], grad_fn=<SubBackward0>)
    b的估计误差: tensor([0.0005], grad_fn=<RsubBackward1>)
"""