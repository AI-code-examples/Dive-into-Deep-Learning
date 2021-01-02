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
@Desc       :   Sec 5.4 池化层
@小结：
1.  最大池化：取池化窗口中输入元素的最大值作为输出
2.  平均池化：取池化窗口中输入元素的平均值作为输出
3.  通过指定池化层的填充和步幅来调整池化的视野
4.  多通道池化时，池化层的输入通道数与输出通道数相同
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    X = nd.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])
    show_title("5.4.1 二维最大池化层")
    print(my_pool2d(X, (2, 2)))
    show_title("5.4.1 二维平均池化层")
    print(my_pool2d(X, (2, 2), 'avg'))
    show_title("5.4.2 填充")
    X = nd.arange(16).reshape((1, 1, 4, 4))
    print("X=", X)
    show_subtitle("MXNet的池化层，使用 3x3 池化窗口，默认填充=0，默认步幅=None")
    pool2d = nn.MaxPool2D(3)
    print(pool2d(X))
    show_subtitle("MXNet的池化层，使用 3x3 池化窗口，指定填充=1，指定步幅=2")
    pool2d = nn.MaxPool2D(3, padding=1, strides=2)
    print(pool2d(X))
    show_subtitle("MXNet的池化层，使用 2x3 池化窗口，指定填充=（1,2)，指定步幅=(2,3)")
    pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
    print(pool2d(X))
    show_title("5.4.3 多通道池化层")
    X = nd.concat(X, X + 1, dim=1)
    print("多通道 X=", X)
    show_subtitle("自动完成多通道池化输出")
    pool2d = nn.MaxPool2D(3, padding=1, strides=2)
    print(pool2d(X))
    pass


def my_pool2d(X, pool_size, mode='max'):
    """
    池化函数
    """
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
                pass
            pass
        pass
    return Y


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
