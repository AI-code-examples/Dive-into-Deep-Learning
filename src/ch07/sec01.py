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
@Desc       :   Sec 7.1 深度学习的优化方法
@小结：
1.  由于优化算法的目标函数通常是一个基于训练数据集的损失函数，优化的目标在于降低训练误差
2.  由于深度学习模型参数通常都是高维的，目标函数的鞍点通常比局部最小值更常见
"""
import d2lzh as d2l
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # plot_mininum_point()
    # plot_saddle_point()
    plot_saddel_point_3d()
    pass


def plot_saddel_point_3d():
    # np.mgrid[a1:b1:c1,a2:b2:c2j,...] 返回多维结构
    # c1 以c1表示间隔；长度为[a,b)，左闭右开
    # c2j 以c2表示数据点个数；长度为[a,b]，左闭右闭
    # 第一维生成的数据按第二个维度横向扩展
    # 第二维生成的数据按第一个维度纵向扩展
    x, y = np.mgrid[-1:1:31j, -1:1:31j]
    z = x ** 2 - y ** 2
    from mpl_toolkits.mplot3d import Axes3D
    ax = d2l.plt.figure().add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
    ax.plot([0], [0], [0], 'rx')
    ticks = [-1, 0, 1]
    d2l.plt.xticks(ticks)
    d2l.plt.yticks(ticks)
    ax.set_zticks(ticks)
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('y')


def plot_saddle_point():
    x = np.arange(-2.0, 2.0, 0.1)
    fig, = d2l.plt.plot(x, x ** 3)
    fig.axes.annotate("saddle point", xy=(0, -0.2), xytext=(-0.52, -5.0), arrowprops=dict(arrowstyle="->"))
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')


def plot_mininum_point():
    d2l.set_figsize((4.5, 2.5))
    x = np.arange(-1.0, 2.0, 0.1)
    fig, = d2l.plt.plot(x, f(x))
    fig.axes.annotate("local minimum", xy=(-0.3, -0.25), xytext=(-0.77, -1.0), arrowprops=dict(arrowstyle="->"))
    fig.axes.annotate("global minimum", xy=(1.1, -0.95), xytext=(0.6, 0.8), arrowprops=dict(arrowstyle="->"))
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')


def f(x):
    return x * np.cos(np.pi * x)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
