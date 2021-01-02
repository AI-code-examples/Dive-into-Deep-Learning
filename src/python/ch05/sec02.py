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
@Desc       :   Sec 5.2 卷积层的填充 和 步幅
@小结：
1.  填充可以增加输出的高和宽
2.  步幅可以减少输出的高和宽
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    show_title("5.2.1 填充")
    # 使用填充保证卷积层的输入和输出的形状相同
    X = nd.random.uniform(shape=(8, 8))
    print("输入（X）的形状=", X.shape)
    conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
    print("卷积计算后的形状=", comp_cov2d(conv2d, X).shape)

    conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
    print("卷积计算后的形状=", comp_cov2d(conv2d, X).shape)

    show_title("5.2.2 步幅")
    conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
    print("卷积计算后的形状=", comp_cov2d(conv2d, X).shape)

    conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
    print("卷积计算后的形状=", comp_cov2d(conv2d, X).shape)

    pass


def comp_cov2d(conv2d, X):
    """
    卷积层计算函数：初始化卷积层权重，对输入和输出进行相应的升维或者降维
    """
    conv2d.initialize()
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 排除不关心的前两个维度：批量和通道
    return Y.reshape(Y.shape[2:])


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
