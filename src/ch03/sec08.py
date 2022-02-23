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
@Desc       :   Sec 3.8 多层感知机
@小结：
1.  多层感知机在输出层与输入层之间加入了一个或者多个全连接隐藏层，并且通过激活函数对隐藏层输出进行变换
2.  常用的激活函数包括：ReLU函数、Sigmoid函数、Tanh函数。
"""
import d2lzh as d2l

from mxnet import autograd, nd
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def main():
    x = nd.arange(-8.0, 8.0, 0.1)
    x.attach_grad()
    with autograd.record():
        y = x.relu()
    xyplot(x, y, 'relu')
    with autograd.record():
        y = x.sigmoid()
    xyplot(x, y, 'sigmoid')
    y.backward()
    xyplot(x, x.grad, 'grad of sigmoid')
    with autograd.record():
        y = x.tanh()
    xyplot(x, y, 'tanh')
    y.backward()
    xyplot(x, x.grad, 'grad of tanh')
    pass


def xyplot(x_vals, y_vals, name):
    d2l.plt.figure()
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    from matplotlib import pyplot as plt

    if len(plt.get_fignums()) > 0:
        plt.show()
