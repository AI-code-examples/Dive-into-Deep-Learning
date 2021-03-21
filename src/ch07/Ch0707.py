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
@Desc       :   Sec
@小结：
"""
import random

from d2l import mxnet as d2l
from mxnet import nd, np, npx, autograd
from mxnet.gluon import nn

from src.ch07.Ch0706 import resnet_block
from tools import beep_end, show_subtitle, show_title, show_figures, net_details

npx.set_np()


# ----------------------------------------------------------------------
def main():
    blk = DenseBlock(2, 10)
    blk.initialize()
    X = np.random.uniform(size=(4, 3, 8, 8))
    Y = blk(X)
    print(Y.shape)

    blk = transition_block(10)
    blk.initialize()
    print(blk(Y).shape)

    # net = dense_net()
    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    net.add(resnet_block(64, 1, first_block=True),
            DenseBlock(4, 32),
            resnet_block(192, 1),
            DenseBlock(4, 32),
            resnet_block(320, 1),
            DenseBlock(4, 32),
            resnet_block(448, 1),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.GlobalAvgPool2D(),
            nn.Dense(10))

    X = np.random.uniform(size=(1, 1, 96, 96))
    net.initialize(force_reinit=True)
    net_details(X, net)

    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    # d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
    pass


def dense_net():
    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        num_channels += num_convs * growth_rate
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(transition_block(num_channels))
    net.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.GlobalAvgPool2D(),
            nn.Dense(10))
    return net


def transition_block(num_channels):
    # 过渡层：控制通道数目，使之不要过大
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk


def conv_block(num_channels):
    # 稠密块：输入和输出是如何连结的
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk


class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = np.concatenate((X, Y), axis=1)
        return X


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
