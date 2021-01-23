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
@Desc       :   Sec 7.7 AdaDelta算法
@小结：
1.  AdaDelta 算法没有学习率超参数，而是通过使用有关自变量更新量平方的指数加权移动平均的项来替代 RMSProp 算法中的学习率。
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # 7.7.1 AdaDelta 算法
    # 在时间步 $t>0$ 的计算上与 RMSProp 一样
    # $s_t\leftarrow\gamma s_{t-1}+(1-\gamma) g_t\odot g_t$
    # 增加状态变量 $\Deleta x_t$，用于更新自变量的变化量
    # $g_t'\leftarrow\sqrt{\frac{\Delta x_{t-1}+\epsilon}{s_t+\epsilon}}\odot g_t$
    # 更新自变量：$x_t\leftarrow x_{t-1}-g_t'=x_{t-1}-\sqrt{\frac{\Delta x_{t-1}+\epsilon}{s_t+\epsilon}}\odot g_t$
    # 使用 $\Delta x_t$ 来记录自变量变化量 $g_t'$ 按元素平方的指数加权移动平均
    # $\Delta x_t\leftarrow \rho\Delta x_{t-1}+(1-\rho) g_t'\odot g_t'$
    # 使用 $\sqrt{\Delta x_{t-1}}$ 来替代 RMSProp 中的超参数 $\eta$
    features, labels = data.get_data_ch7()

    def init_adadelta_states():
        s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
        delta_w, delta_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
        return (s_w, delta_w), (s_b, delta_b)

    def adadelta(params, states, hyperparams):
        rho, eps = hyperparams['rho'], 1e-5
        for p, (s, delta) in zip(params, states):
            s[:] = rho * s + (1 - rho) * p.grad.square()
            g = ((delta + eps).sqrt() / (s + eps).sqrt()) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g

    d2l.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)

    d2l.train_gluon_ch7('adadelta', {'rho': 0.9}, features, labels)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
