"""
=================================================
@path   : Dive-into-Deep-Learning -> sec0302.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-23 15:22
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   3.2.线性回归的原始实现
==================================================
"""
import random

import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

from tools import beep_end
from tools import show_subtitle


# ----------------------------------------------------------------------
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def main():
    # 3.2.1 生成数据集
    num_inputs = 2
    num_examples = 1000
    features, labels = create_dataset(num_examples)
    print("features[0]=", features[0])
    print("labels[0]=", labels[0])
    # 绘制数据集的散点图
    paint_scatter(features, labels)
    # 3.2.2 读取数据集
    read_dataset(features, labels)
    # 3.2.3 初始化模型参数
    pred_w, pred_b = train_model(features, labels)
    show_subtitle("pred_w=")
    print(pred_w)
    show_subtitle("pred_b=")
    print(pred_b)
    pass


def train_model(features, labels):
    num_inputs = features.ndim
    w, b = initial_model(num_inputs)
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    batch_size = 10
    show_subtitle("training")
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print("epoch %d, loss %f" % (epoch + 1, train_l.mean().numpy()))
    return w, b


def initial_model(num_inputs):
    """初始化模型参数"""
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b


def read_dataset(features, labels):
    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        show_subtitle(" X ")
        print(X)
        show_subtitle(" y ")
        print(y)
        break


def paint_scatter(features, labels):
    d2l.set_figsize()
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)


def synthetic_data(w, b, num_examples):
    """生成 $y=Xw+b+噪声$ 的数据"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def create_dataset(num_examples):
    true_w = torch.tensor([2, -3.4])  # 生成样本集的参数
    true_b = 4.2  # 生成样本集的偏差
    print("true_w", true_w, end="\t")
    print("true_b", true_b)
    features, labels = synthetic_data(true_w, true_b, num_examples)
    return features, labels


def linreg(X, w, b):
    # 定义线性模型
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    # 定义损失函数
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """定义优化算法:小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
    if len(plt.get_fignums()) > 0:
        plt.show()
