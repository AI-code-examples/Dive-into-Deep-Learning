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
@Desc       :   Sec 5.7 使用重复元素的网络（VGG）
@小结：
1.  VGG-11 通过 5 个可以重复使用的卷积块来构造网络
2.  VGG 的卷积块里面卷积层的个数和输出通道数可以自定义，从而定义出不同的 VGG 模型
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    conv_arch = (
        # 两个单卷积层
        (1, 64),
        (1, 128),
        # 三个双卷积层
        (2, 256),
        (2, 512),
        (2, 512)
    )
    ctx = d2l.try_gpu()
    # toy_vgg(conv_arch, ctx)
    # 增加卷积层的通道数（ratio=2)，减少全连接层的「神经元个数（2048）」和「batch_size=64」依然可以增加收敛的速度和精度
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    # toy_vgg(small_conv_arch, ctx)
    mnist_vgg(small_conv_arch, ctx)
    pass


def mnist_vgg(conv_arch, ctx):
    lr, num_epochs, batch_size = 0.05, 5, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    net = vgg(conv_arch)
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
    pass


def toy_vgg(conv_arch, ctx):
    X = nd.random.uniform(shape=(1, 1, 224, 224), ctx=ctx)
    print("X.shape:\t", X.shape)
    net = vgg(conv_arch)
    net.initialize(ctx=ctx)
    for blk in net:
        X = blk(X)
        print(blk.name, "output shape:\t", X.shape)
        pass
    pass


def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk


def vgg(conv_arch):
    net = nn.Sequential()
    # 卷积层部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
        pass
    # 全连接层部分
    net.add(
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5)
    )
    net.add(nn.Dense(10))
    return net


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
