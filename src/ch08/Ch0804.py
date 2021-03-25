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
@Desc       :   Sec
@小结：
"""
import random

from d2l import mxnet as d2l
from mxnet import nd, np, npx, autograd
from mxnet.gluon import nn
from tools import beep_end, show_subtitle, show_title, show_figures

npx.set_np()


# ----------------------------------------------------------------------
def main():
    # 8.4.2 Recurrent Neural Networks with Hidden States
    X, W_xh = np.random.normal(0, 1, (3, 1)), np.random.normal(0, 1, (1, 4))
    H, W_hh = np.random.normal(0, 1, (3, 4), np.random.normal(0, 1, (4, 4)))
    np.dot(X, W_xh) + np.dot(H, W_hh)
    np.dot(np.concatenate((X, H), 1), np.concatenate((W_xh, W_hh), 0))

    # 8.4.3 RNN-based Character-Level Language Models
    # 8.4.4 Perplexity

    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
