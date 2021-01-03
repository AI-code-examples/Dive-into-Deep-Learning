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
@Desc       :   Sec 5.9 包含并行连结的网络（GoogLeNet）
@小结：
1.  Inception 块相当于一个有着 4 条线路的子网络。
因此通过不同窗口形状的卷积层和最大池化层来并行抽取信息，并且使用 1x1 卷积层减少通道数从而降低模型的复杂度
2.  GoogLeNet 将多个设计精细的 Inception 块和其他层串联起来。
其中，Inception 块的通道数分配比例是在 ImageNet 数据集上通过大量的实验得来
3.  GoogLeNet 与其后继者一度是 ImageNet 上最高效的模型之一：
即在类似的测试精度下，它们的计算复杂度更低
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    net = create_net()
    data_size = 96
    toy_googlenet(data_size, net)
    mnist_googlenet(data_size, net)
    pass


def mnist_googlenet(data_size, net):
    lr, num_epochs, batch_size, ctx = 0.1, 25, 128, d2l.try_gpu()
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=data_size)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def toy_googlenet(data_size, net):
    X = nd.random.uniform(shape=(1, 1, data_size, data_size))
    print("X.shape:\t", X.shape)
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, "output shape:\t", X.shape)
        pass


def create_net():
    # 第一个模块：64 通道的 7x7 卷积层
    b1 = nn.Sequential()
    b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    # 第二个模块：64 通道的 1x1 卷积层 + 192 通道的 3x3 卷积层
    b2 = nn.Sequential()
    b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
           nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    # 第三个模块：Inception 块（64+128+32+32=256）通道（2:4:1:1）+ Inception块（128+192+96+64=480）通道（4:6:3:2）
    b3 = nn.Sequential()
    b3.add(Inception(64, (96, 128), (16, 32), 32),
           Inception(128, (128, 192), (32, 96), 64),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    # 第四个模块：Inception 块（192+208+48+64=512）+ Inception 块（160+224+64+64=512）+ Inception 块（128+256+64+64=512）
    #               + Inception 块（112+288+64+64=528）+ Inception 块（256+320+128+128=832）
    b4 = nn.Sequential()
    b4.add(Inception(192, (96, 208), (16, 48), 64),
           Inception(160, (112, 224), (24, 64), 64),
           Inception(128, (128, 256), (24, 64), 64),
           Inception(112, (144, 288), (32, 64), 64),
           Inception(256, (160, 320), (32, 128), 128),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    # 第五个模块：Inception 块（256+320+128+128=832）+ Inception 块（384+384+128+128=1024）
    b5 = nn.Sequential()
    b5.add(Inception(256, (160, 320), (32, 128), 128),
           Inception(384, (192, 384), (48, 128), 128),
           nn.GlobalAvgPool2D())
    net = nn.Sequential()
    net.add(b1, b2, b3, b4, b5, nn.Dense(10))
    return net


class Inception(nn.Block):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单 1x1 卷积层
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # 线路2，1x1 卷积层 + 3x3 卷积层
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1, activation='relu')
        # 线路3，1x1 卷积层 + 5x5 卷积层
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2, activation='relu')
        # 线路4，3x3 最大池化层 + 1x1 卷积层
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')
        pass

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维上连结输出
        return nd.concat(p1, p2, p3, p4, dim=1)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
