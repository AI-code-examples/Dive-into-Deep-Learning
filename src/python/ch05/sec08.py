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
@Desc       :   Sec 5.8 网络中的网络（NiN）
@小结：
1.  NiN 重复使用由卷积层和代替全连接层的 1x1 卷积层构成的 NiN 块来构建深层网络
2.  NiN 去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层
3.  NiN 串联多个由卷积层和「全连接层」构成的小网络来构建一个深层网络
4.  NiN 的设计思想影响了后面一系列卷积神经网络的设计
5.  NiN 的精度没有 VGG 更好
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    net = create_net()
    ctx = d2l.try_gpu()
    toy_nin(ctx, net)
    mnist_nin(ctx, net)
    pass


def create_net():
    net = nn.Sequential()
    net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2D(pool_size=3, strides=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2D(pool_size=3, strides=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Dropout(0.5),
            # 标签类别数是 10
            nin_block(10, kernel_size=3, strides=1, padding=1),
            # 全局平均池化层将窗口形状自动设置成输入的高和宽
            nn.GlobalAvgPool2D(),
            # 将四维的输出转成二维的输出，其形状为（batch_size,10)
            nn.Flatten()
            )
    return net


def mnist_nin(ctx, net):
    lr, num_epochs, batch_size = 0.05, 15, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def toy_nin(ctx, net):
    X = nd.random.uniform(shape=(2, 1, 224, 224), ctx=ctx)
    print("X.shape:\t", X.shape)
    net.initialize(ctx=ctx)
    for layer in net:
        X = layer(X)
        print(layer.name, "output shape:\t", X.shape)
        pass


def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu')
            )
    return blk


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
