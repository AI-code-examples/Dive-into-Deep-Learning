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
@Desc       :   Sec 3.14 正向传播、反向传播、计算图
@小结：
1.  正向传播沿着输入层到输出层的顺序，依次计算并且存储神经网络的中间变量
2.  反向传播沿着输出层到输入层的顺序，依次计算并且存储神经网络的中间变量和参数的梯度
3.  在训练深度学习模型时，正向传播和反向传播相互依赖
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
