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
@Desc       :   Sec 5.3 多个输入通道 和 多个输出通道
@小结：
1.  使用多通道可以拓展卷积层的模型参数
2.  假设将通道维当作特征维，将高和宽维度上的元素当成数据样本，那么 1x1 卷积层的作用与全连接层等价
3.  1x1 卷积层通常用来调整网络层之间的通道数，还可以用于控制模型的复杂度
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    show_title("5.3.1 多个输入通道")
    X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = nd.array([[[0, 1], [2, 3]],
                  [[1, 2], [3, 4]]])
    print("原始的 K 的形状=", K.shape)
    print("多输入单输出计算的结果=", corr2d_multi_in(X, K))

    show_title("5.3.2 多个输出通道")
    K = nd.stack(K, K + 1, K + 2)
    print("堆叠后的 K 的形状=", K.shape)
    print("多输入多输出计算的结果=", corr2d_multi_in_out(X, K))

    show_title("5.3.3 卷积层的形状为 1x1")
    X = nd.random.uniform(shape=(3, 3, 3))
    K = nd.random.uniform(shape=(2, 3, 1, 1))
    Y1 = corr2d_multi_in_out(X, K)
    Y2 = corr2d_multi_in_out_1x1(X, K)
    print((Y1 - Y2).norm().asscalar())
    pass


def corr2d_multi_in(X, K):
    """
    多个输入通道单个输出通道的互相关运算函数
    """
    # 首先沿着 X 和 K 的第 0 维（即通道维）遍历
    # 然后使用 * 将结果列表变成 add_n() 函数的位置参数来进行相加
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])


def corr2d_multi_in_out(X, K):
    """
    多个输入通道多个输出通道的互相关运算函数
    """
    # 首先沿着 K 的第 0 维（即通道维）遍历
    # 然后同输入 X 做互相关运算
    # 最后将所有的结果使用 stack() 函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)  # 全连接层的矩阵乘法
    return Y.reshape((c_o, h, w))


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
