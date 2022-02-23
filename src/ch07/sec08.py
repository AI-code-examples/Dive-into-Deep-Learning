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
@Desc       :   Sec 7.8 Adam 算法
@小结：
1.  Adam 算法在 RMSProp 算法的基础上对小批量随机梯度也做了指数加权移动平均
2.  Adam 算法使用了偏差修正
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
    # Adam 算法在 RMSProp 算法基础上对小批量随机梯度做了指数加权移动平均
    # Adam 算法使用了动量变量 $v_t$ 和 RMSProp 算法中小批量随机梯度按元素平方的指数加权移动平均变量 $s_t$
    # 超参数 $\beta_1=0.9$，时间步 $t$ 的动量变量 $v_t$ 即小批量随机梯度 $g_t$ 的指数加权移动平均
    # $v_t\leftarrow\beta_1 v_{t-1}+(1-\beta_1)g_t$
    # 与 RMSProp 算法一样，超参数 $beta_2=0.999$，小批量随机梯度按元素平方后的项 $g_t\odot g_t$ 做指数加权移动平均得到 $s_t$
    # $s_t\leftarrow\beta_2 s_{t-1}+(1-\beta_2) g_t\odot g_t$
    # 当 $t$ 较小时，过去各个时间步小批量随机梯度权值之和也会较小
    # 为了消除这样的影响，对于任意时间步 $t$，可以除以 $1-\beta_{?}^t$，从而使过去各时间步小批量随机梯度权值之和为1，这个操作叫做偏差修正
    # $\hat{v}_t\leftarrow\frac{v_t}{1-\beta_1^t}$
    # $\hat{s}_t\leftarrow\frac{s_t}{1-\beta_2^t}$
    # 使用修正后的变量更新随机梯度，目录函数自变量中的每个元素都分别拥有自己的学习率
    # $g_t'\leftarrow\frac{\eta\hat{v}_t}{\sqrt{\hat{s}_t}+\epsilon}$
    # 使用 $g_t'$ 更新自变量
    # $x_t\leftarrow x_{t-1}-g_t'$
    features, labels = data.get_data_ch7()

    def init_adam_states():
        v_w, v_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
        s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
        return (v_w, s_w), (v_b, s_b)

    def adam(params, states, hyperparams):
        beta1, beta2, eps = 0.9, 0.999, 1e-6
        learning_rate = hyperparams['lr']
        for p, (v, s) in zip(params, states):
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * p.grad.square()
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= learning_rate * v_bias_corr / (s_bias_corr.sqrt() + eps)
            pass
        hyperparams['t'] += 1
        pass

    d2l.train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)
    d2l.train_gluon_ch7('adam', {'learning_rate': 0.01}, features, labels)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
