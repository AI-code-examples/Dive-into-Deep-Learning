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
@Desc       :   Sec 4.6 GPU 计算
@备注：
1.  因为我的机器只有一个 GPU，因此修改了部分代码
2.  环境中需要安装 MXNet 的 GPU 版本，详情参考 readme.md
@小结：
"""
import d2lzh as d2l
import mxnet as mx

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # calc_in_gpu()
    show_title("4.6.3 Gluon 的 GPU 计算")
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(ctx=mx.gpu())
    X = nd.array([1, 2, 3], ctx=mx.gpu())
    print(net(X))
    show_subtitle("模型的权重")
    print(net[0].weight.data())
    pass


def calc_in_gpu():
    show_title("4.6.2 GPU 上的计算")
    show_subtitle("GPU上的存储")
    x = nd.array([1, 2, 3])
    print('x=', x)
    y = x.copyto(mx.gpu())
    print('y=', y)
    print("x is y", x is y)
    y1 = y.as_in_context(mx.gpu())
    print("y1 is y", y1 is y)
    y2 = y.copyto(mx.gpu())
    print("y2 is y", y2 is y)
    z = x.as_in_context(mx.gpu())
    print('z=', z)
    print("x is z", x is z)
    a = nd.array([1, 2, 3], ctx=mx.gpu())
    print('a=', a)
    show_subtitle("GPU上的计算")
    print((z + 2).exp() * y)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
