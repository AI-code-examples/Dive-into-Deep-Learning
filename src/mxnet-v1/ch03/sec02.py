# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   Dive-into-Deep-Learning
@File       :   sec0202.py
@Version    :   v0.1
@Time       :   2020-12-27 9:25
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   Sec 3.2 线性回归的实现
@理解：
"""
import random

from IPython import display
from mxnet import nd, autograd
from tools import beep_end, show_subtitle
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------
def use_svg_display():
    # 用失量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize
    pass


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  # take() 根据索引返回对应元素


def main():
    # 3.2.1 生成数据集
    num_inputs = 2
    num_examples = 1000
    features, labels = create_dataset(num_examples, num_inputs)
    # print("features[0]=", features[0])
    # print("labels[0]=", labels[0])
    # 绘制数据集的散点图
    # paint_scatter(features, labels)
    # 3.2.2 读取数据集
    # read_dataset(features, labels)
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
            with autograd.record():
                l = loss(net(X, w, b), y)
            l.backward()
            sgd([w, b], lr, batch_size)
        train_l = loss(net(features, w, b), labels)
        print("epoch %d, loss %f" % (epoch + 1, train_l.mean().asnumpy()))
    return w, b


def initial_model(num_inputs):
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
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
    set_figsize()
    plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)


def create_dataset(num_examples, num_inputs):
    true_w = [2, -3.4]  # 生成样本集的参数
    true_b = 4.2  # 生成样本集的偏差
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    print("true_w", true_w, end="\t")
    print("true_b", true_b)
    return features, labels


def linreg(X, w, b):
    # 定义线性模型
    return nd.dot(X, w) + b


def squared_loss(y_hat, y):
    # 定义损失函数
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    # 定义优化算法
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
    if len(plt.get_fignums()) > 0:
        plt.show()
