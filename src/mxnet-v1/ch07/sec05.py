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
@Desc       :   Sec 7.5 AdaGrad 算法
@小结：
1.  AdaGrad 算法在迭代过程中不断调整学习率
2.  AdaGrad 算法允许不同维度迭代过程中拥有不同的学习率
3.  使用 AdaGrad 算法时，自变量中每个元素的学习率在迭代过程中会一直减小或者不变
"""
import math

import d2lzh as d2l
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

import data
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # adagrad_func()
    # adagrad_method()
    features, labels = data.get_data_ch7()
    d2l.train_gluon_ch7('adagrad', {'learning_rate': 0.1}, features, labels)
    pass


def adagrad_method():
    features, labels = data.get_data_ch7()

    def init_adagrad_states():
        s_w = nd.zeros((features.shape[1], 1))
        s_b = nd.zeros(1)
        return s_w, s_b

    def adagrad(params, states, hyperparams):
        eps = 1e-6
        for p, s in zip(params, states):
            s[:] += p.grad.square()
            p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()
            pass
        pass

    d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)


def adagrad_func():
    # 7.5.2 算法实现
    def adagrad_2d(x1, x2, s1, s2):
        # $g_t\odot g_t$：小批量随机梯度 $g_t$ 按元素平方
        # $s_t\leftarrow s_{t-1} + g_t\odot g_t, s_0=0$：累加变量 $s_t$
        # $x_t\leftarrow x_{t-1} - \frac{\eta}{\sqrt{s_t+\epsilon}}\odot g_t$
        # $\eta$ 是学习率；$\epsilon$ 是维持数值稳定性，避免分母为零而添加的常数，一般取 1e-6
        g1, g2 = 0.2 * x1, 4 * x2  # 自变量的梯度
        eps = 1e-6
        s1 += g1 ** 2
        s2 += g2 ** 2
        x1 -= eta / math.sqrt(s1 + eps) * g1
        x2 -= eta / math.sqrt(s2 + eps) * g2
        return x1, x2, s1, s2

    def f_2d(x1, x2):
        return 0.1 * x1 ** 2 + 2 * x2 ** 2

    eta = 0.5
    d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
    eta = 2
    d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
