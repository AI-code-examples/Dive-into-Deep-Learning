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
@Desc       :   Sec 5.1 二维卷积层
@小结：
1.  二维卷积层的核心计算是二维互相关运算
2.  二维卷积层的输出=二维输入数据和卷积核做互相关运算+偏差
3.  使用卷积核可以检测图像中的边缘
4.  可以基于输入的数据学习卷积核
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # test_corr2d()
    # test_edge()
    conv2d = nn.Conv2D(1, kernel_size=(1, 2))
    conv2d.initialize()
    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    K = nd.array([[1, -1]])
    Y = corr2d(X, K)
    # 二维卷积输入的格式为：（样本，通道，样本，样本）
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape(1, 1, 6, 7)

    for i in range(10):
        with autograd.record():
            Y_hat = conv2d(X)
            l = (Y_hat - Y) ** 2
            pass
        l.backward()
        conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
        if (i + 1) % 2 == 0:
            print("batch %d, loss %.3f" % (i + 1, l.sum().asscalar()))
            pass
        pass

    show_subtitle("Kernel Matrix")
    print(conv2d.weight.data())
    pass


def test_edge():
    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    print("X=", X)
    K = nd.array([[1, -1]])
    Y = corr2d(X, K)
    print("Y=", Y)


def test_corr2d():
    X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = nd.array([[0, 1], [2, 3]])
    print(corr2d(X, K))


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1, 1))
        pass

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 对单个元素 Y[i,j] 赋值，会导致无法求梯度
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
            pass
        pass
    return Y


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
