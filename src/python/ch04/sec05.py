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
@Desc       :   Sec 4.5 模型的保存和恢复
@小结：
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_figures, show_title


# ----------------------------------------------------------------------
def main():
    # save_load_ndarray()
    show_title("4.5.2 Gluon 模型参数的读写")
    net = MLP()
    net.initialize()
    X = nd.random.uniform(shape=(2, 20))
    y = net(X)
    print("第一个模型的训练结果：",y)
    filename = 'mlp.params'
    net.save_parameters(filename)

    net2 = MLP()
    net2.load_parameters(filename)
    y2 = net2(X)
    print("第二个模型的训练结果：",y2)

    print("输出结果相等=", (y == y2))
    pass


class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)
        pass

    def forward(self, x):
        return self.output(self.hidden(x))

    pass


def save_load_ndarray():
    show_title("4.5.1 NDArray 的读写")
    x1 = nd.ones(3)
    print('x1=', x1)
    nd.save('x1', x1)
    x2 = nd.load('x1')
    print('x2=', x2)
    y1 = nd.zeros(4)
    print('x1=', x1)
    print('y1=', y1)
    nd.save('x1y1', [x1, y1])
    x2, y2 = nd.load('x1y1')
    print('x2=', x2)
    print('y2=', y2)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
