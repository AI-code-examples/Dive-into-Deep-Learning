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
@Desc       :   Sec 7.2 梯度下降和随机梯度下降
@小结：
1.  使用适当的学习率，沿着梯度反方向更新自变量可能降低目标函数值。
2.  随着梯度下降重复更新过程直到满足要求的解
3.  学习率过大会造成学习结果发散；学习率过小导致学习速度过慢
4.  合适的学习率需要通过多次实验找到
5.  当训练数据集的样本较多时，梯度下降每次迭代的计算开销较大，因而随机梯度下降更受关注
"""
import d2lzh as d2l
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # show_trace(gd(0.2))
    # show_trace(gd(0.05))
    # show_trace(gd(1.1))
    # show_trace_2d(f_2d, train_2d(gd_2d, 0.1))
    show_trace_2d(f_2d, train_2d(sgd_2d, 0.1))
    pass


def sgd_2d(x1, x2, s1, s2, eta):
    return x1 - eta * (2 * x1 + np.random.normal(0.1)), x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0


def gd_2d(x1, x2, s1, s2, eta):
    return x1 - eta * 2 * x1, eta * 4 * x2, 0, 0


def f_2d(x1, x2):
    """
    目标函数
    """
    return x1 ** 2 + 2 * x2 ** 2


def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')


def train_2d(trainer, eta):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2, eta)
        results.append((x1, x2))
        print("epoch %d, x1 %f, x2 %f" % (i + 1, x1, x2))
        pass
    return results


def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    d2l.set_figsize()
    d2l.plt.plot(f_line, [x * x for x in f_line])
    d2l.plt.plot(res, [x * x for x in res], "-o")
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    return


def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        # f(x)=x*x 的层数为 f'(x)=2*x
        x -= eta * 2 * x
        results.append(x)
        pass
    print("epoch 10,x:", x)
    return results


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
