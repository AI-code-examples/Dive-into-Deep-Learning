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
@Desc       :   Sec 3.15 数值稳定性 和 模型初始化
@小结：
1.  深度学习模型的数值稳定性主要面临两个问题：「衰减」和「爆炸」。当神经网络的层数过多时，模型的数值稳定性容易变差
2.  通常随机初始化模型的参数
    1.  MXNet 默认采用随机初始化：权重参数随机采样于(-0.07,0.07)的均匀分布，偏差参数全部清零
    2.  Xavier 随机初始化：权重参数随机采样于 $(-\sqrt{ 6/(a+b) },\sqrt{ 6/(a+b) })$ 的均匀分布
        -   设计思想：模型参数初始化后，每层输出的方差不受该输入个数的影响，每层梯度的方差不受该层输入个数的影响
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_figures


# ----------------------------------------------------------------------
def main():
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
