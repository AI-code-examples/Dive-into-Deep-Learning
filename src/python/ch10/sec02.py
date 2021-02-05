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
@Desc       :   Sec 10.2 近似训练
@小结：
1.  负采样通过考虑同时含有正类样本和负类样本的相互独立事件来构造损失函数。其训练中每一步的梯度计算开销与采样的噪声词的个数线性相关
2.  层序 softmax 使用了二叉树，并且根据根结点到叶结点的路径来构造损失函数，其训练中每一步的梯度计算开销与词典大小的对数相关。
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
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
