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
@Desc       :   Sec 6.2 循环神经网络
@小结：
1.  使用循环计算的网络即循环神经网络
2.  循环神经网络的隐藏状态可以捕捉截至当前时间步的序列的历史信息
3.  循环神经网络模型参数的数量不随时间步的增加而增长
4.  可以基于字符级循环神经网络来创建语言模型
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    X, W_xh = nd.random.uniform(shape=(3, 1)), nd.random.uniform(shape=(1, 4))
    H, W_hh = nd.random.normal(shape=(3, 4)), nd.random.normal(shape=(4, 4))
    print(nd.dot(X, W_xh) + nd.dot(H, W_hh))
    print(nd.dot(nd.concat(X, H, dim=1), nd.concat(W_xh, W_hh, dim=0)))
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
