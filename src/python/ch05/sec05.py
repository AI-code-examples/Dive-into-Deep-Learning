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
@Desc       :   Sec 5.5 卷积神经网络（LeNet）
@小结：
1.  卷积神经网络就是含有卷积层的网络
2.  LeNet 就是交替使用卷积层和最大池化层，输出使用全连接层，来进行图像分类的网络模型
"""
import d2lzh as d2l
import mxnet as mx
import time

from mxnet import autograd, gluon, init, nd, ndarray
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # toy_lenet()
    # 2060 比 i7-9700 快 4倍
    mnist_lenet()
    pass


def mnist_lenet():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.9, 5
    ctx = try_gpu()
    print(ctx)
    net = create_net()
    # ToDo：为什么不使用 Xavier() 初始化，训练结果就不会收敛呢？
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def toy_lenet():
    ctx = try_gpu()
    X = nd.random.uniform(shape=(1, 1, 28, 28), ctx=ctx)
    print("X.shape:\t", X.shape)
    net = create_net()
    net.initialize(ctx=ctx)
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)


def create_net():
    net = nn.Sequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120, activation='sigmoid'),
            nn.Dense(84, activation='sigmoid'),
            nn.Dense(10))
    return net


def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    print("training on", ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
                pass
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
            pass
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec"
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))


def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
        pass
    return acc_sum.asscalar() / n


def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
        pass
    return ctx


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
