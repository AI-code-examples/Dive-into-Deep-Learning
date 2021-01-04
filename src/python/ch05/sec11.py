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
@Desc       :   Sec 5.11 残差网络（ResNet）
@小结：
1.  残差块通过跨层的数据通道从而能够训练出更加有效，也更有深度的深度神经网络
2.  ResNet 深刻影响了后来的深度神经网络的设计
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # test_residual_block()
    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            )
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(),
            nn.Dense(10)
            )
    data_size = 96
    X = nd.random.uniform(shape=(1, 1, data_size, data_size))
    print("X.shape:\t", X.shape)
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, "output shape:\t", X.shape)
        pass
    lr, num_epochs, batch_size, ctx = 0.05, 15, 96, d2l.try_gpu()
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=data_size)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
    pass


def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1cov=True, strides=2))
        else:
            blk.add(Residual(num_channels))
            pass
        pass
    return blk


def test_residual_block():
    X = nd.random.uniform(shape=(4, 3, 6, 6))
    show_subtitle("5.11.1 残差块")
    show_subtitle("输入与输出的形状相同")
    blk = Residual(3)
    blk.initialize()
    print("X.shape=", X.shape, "blk(X).shape=", blk(X).shape)
    show_subtitle("输出通道数加倍，输出维度减半")
    blk = Residual(6, use_1x1cov=True, strides=2)
    blk.initialize()
    print("X.shape=", X.shape, "blk(X).shape=", blk(X).shape)


class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1cov=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1cov:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        pass

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
            pass
        return nd.relu(Y + X)

    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
