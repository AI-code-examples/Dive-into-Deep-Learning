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
@Desc       :   Sec 7.4 动量法
@小结：
1.  动量法使用了指数加权移动平均的思想：将过去时间步的梯度做了加权平均，且权重按时间步指数衰减。
2.  动量法使得相邻时间步的自变量更新在方向上更加一致
"""
import d2lzh as d2l
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

import data
from src.python.ch07.sec02 import show_trace_2d, train_2d
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def momentum_method():
    # 7.4.2 动量法
    # $v_t\leftarrow\gamma v_{t-1}+\eta_t g_t$
    # $x_t\leftarrow x_{t-1}-v_t$
    # 动量超参数 $\gamma\in[0,1)$
    # 1. 指数加权移动平均：当 $\gamma=0.95$ 时，近似20个时间步的值的加权平均；当 $\gamma=0.9$ 时，近似10个时间步的值的加权平均；当当前时间步越近的值获得的权重越大
    # 2. 由指数加权移动平均理解动量法：每个时间步的自变量更新量近似于最近 $\frac{1}{1-\gamma}$ 个时间步的更新量做了指数加权移动平均后再除以 $(1-\gamma)$
    def momentum_2d(x1, x2, v1, v2):
        v1 = gamma * v1 + eta * 0.2 * x1
        v2 = gamma * v2 + eta * 4 * x2
        return x1 - v1, x2 - v2, v1, v2

    eta = 0.4
    eta = 0.6
    gamma = 0.5
    d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
    pass


def sgd_method():
    def gd_2d(x1, x2, s1, s2):
        return x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0

    # 7.4.1 梯度下降法存在的问题
    eta = 0.4
    eta = 0.6
    d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
    pass


def momentum_trainer():
    features, labels = data.get_data_ch7()

    def init_momentum_states():
        v_w = nd.zeros((features.shape[1], 1))
        v_b = nd.zeros(1)
        return v_w, v_b

    def sgd_momentum(params, states, hyperparams):
        for p, v in zip(params, states):
            v[:] = hyperparams['momentum'] * v + hyperparams['lr'] * p.grad
            p[:] -= v
            pass
        pass

    # momentum=0.5, 随机梯度为最近2个时间步的2倍小批量梯度的加权平均
    d2l.train_ch7(sgd_momentum, init_momentum_states(), {'lr': 0.02, 'momentum': 0.5}, features, labels)
    # momentum=0.9，随机梯度为最近10个时间步的10倍小批量梯度的加权平均
    d2l.train_ch7(sgd_momentum, init_momentum_states(), {'lr': 0.02, 'momentum': 0.9}, features, labels)
    # 将学习率减小到 1/5，则10倍小批量梯度比2倍小批量梯度快5倍的速度造成的不平滑问题得到解决
    d2l.train_ch7(sgd_momentum, init_momentum_states(), {'lr': 0.004, 'momentum': 0.9}, features, labels)
    # 使用 Gluon 的简洁实现
    d2l.train_gluon_ch7('sgd', {'learning_rate': 0.004, 'momentum': 0.9}, features, labels)
    pass


def main():
    # sgd_method()
    # momentum_method()
    momentum_trainer()
    pass


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
