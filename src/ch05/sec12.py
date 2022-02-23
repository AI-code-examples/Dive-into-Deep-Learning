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
@Desc       :   Sec 5.12 稠密连接网络（DenseNet）
@小结：
1.  在跨层连接上，ResNet将输入与输出相加；DenseNet将输入与输出在通道维上连结
2.  DenseNet 的主要构建模块是稠密块和过渡层
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # test_transition_block()
    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    # num_channels 为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(transition_block(num_channels))
            pass
        pass
    net.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.GlobalAvgPool2D(),
            nn.Dense(10)
            )
    lr, num_epochs, batch_size, ctx = 0.1, 15, 256, d2l.try_gpu()
    net.initialize(init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
    pass


def test_transition_block():
    blk = DenseBlock(2, 10)
    blk.initialize()
    X = nd.random.uniform(shape=(4, 3, 8, 8))
    Y = blk(X)
    print("Y.shape=", Y.shape)
    blk = transition_block(10)
    blk.initialize()
    print("blk(Y).shape=", blk(Y).shape)


def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk


def conv_block(num_channels):
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
            pass
        pass

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 基于通道维将输入和输出连接在一起
            X = nd.concat(X, Y, dim=1)
            pass
        return X


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
