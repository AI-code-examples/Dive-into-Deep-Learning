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
@Desc       :   Sec 3.3 线性回归的简洁实现
@小结：
-   使用 Gluon 可以更加简洁地实现模型
-   在 Gluon 中
    -   data 模块提供了有关数据处理的工具
    -   nn 模块定义了神经网络的层
    -   loss 模块定义了损失函数
-   MXNet 的 initializer 模块提供了模型参数初始化的各种方法
"""
from mxnet import nd, autograd
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def main():
    features, labels = create_data_set()

    batch_size, data_iter = read_dataset(features, labels)

    net = define_model()

    init_model(net)

    loss = define_loss()

    trainer = define_trainer(net)

    train_model(batch_size, data_iter, features, labels, loss, net, trainer)
    pass


def train_model(batch_size, data_iter, features, labels, loss, net, trainer):
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        print("epoch %d, loss %f" % (epoch, l.mean().asnumpy()))


def define_trainer(net):
    from mxnet import gluon
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
    return trainer


def define_loss():
    from mxnet.gluon import loss as gloss
    loss = gloss.L2Loss()
    return loss


def init_model(net):
    from mxnet import init
    net.initialize(init.Normal(sigma=0.01))


def define_model():
    from mxnet.gluon import nn
    net = nn.Sequential()
    net.add(nn.Dense(1))
    return net


def read_dataset(features, labels):
    from mxnet.gluon import data as gdata
    batch_size = 10
    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    return batch_size, data_iter


def create_data_set():
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    return features, labels


# ----------------------------------------------------------------------


if __name__ == '__main__':
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2

    main()
    # 运行结束的提醒
    beep_end()
