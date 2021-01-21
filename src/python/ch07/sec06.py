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
@Desc       :   Sec 7.6 RMSProp 算法
@小结：
1.  RMSProp算法使用了小批量随机梯度按元素平方的指数加权移动平均来调整学习率
"""
import d2lzh as d2l
import math
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

import data
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # rmsprop_method()
    rmsprop_trainer()
    pass


def rmsprop_trainer():
    features, labels = data.get_data_ch7()

    def init_rmsprop_states():
        s_w = nd.zeros((features.shape[1], 1))
        s_b = nd.zeros(1)
        return s_w, s_b

    def rmsprop(params, states, hyperparams):
        gamma, eps = hyperparams['gamma'], 1e-6
        for p, s in zip(params, states):
            s[:] = gamma * s + (1 - gamma) * p.grad.square()
            p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()

    d2l.train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'gamma': 0.9}, features, labels)
    d2l.train_gluon_ch7('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9}, features, labels)
    pass


def rmsprop_method():
    # $s_t\leftarrow\gamma s_{t-1}+(1-\gamma) g_t\odot g_t$
    # $x_t\leftarrow x_{t-1}-\frac{\eta}{\sqrt{s_t+\epsilon}}\odot g_t$
    # 状态变量是对平方项 $g_t\odot g_t$ 的指数加权移动平均，
    # 即最近 $1/(1-\gamma)$ 个时间步的小批量随机梯度平方项的加权平均，
    # 因此，自变量每个元素的学习率在迭代过程中就不会一直降低或者不变
    def rmsprop_2d(x1, x2, s1, s2):
        g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
        s1 = gamma * s1 + (1 - gamma) * g1 ** 2
        s2 = gamma * s2 + (1 - gamma) * g2 ** 2
        x1 -= eta / math.sqrt(s1 + eps) * g1
        x2 -= eta / math.sqrt(s2 + eps) * g2
        return x1, x2, s1, s2

    def f_2d(x1, x2):
        return 0.1 * x1 ** 2 + 2 * x2 ** 2

    eta, gamma = 0.4, 0.9
    d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
