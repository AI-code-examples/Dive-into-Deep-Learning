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
from tools import beep_end, show_subtitle, show_title, show_figures, net_details

npx.set_np()


# ----------------------------------------------------------------------
def main():
    blk = Residual(3)
    blk.initialize()
    X = np.random.uniform(size=(4, 3, 6, 6))
    print(blk(X).shape)

    blk = Residual(6, use_1x1conv=True, strides=2)
    blk.initialize()
    print(blk(X).shape)

    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(10))

    X = np.random.uniform(size=(1, 1, 224, 224))
    net.initialize()
    net_details(X, net)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
    pass


class Residual(nn.Block):
    """The Residual block of ResNet."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3: X = self.conv3(X)
        return npx.relu(Y + X)


def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
