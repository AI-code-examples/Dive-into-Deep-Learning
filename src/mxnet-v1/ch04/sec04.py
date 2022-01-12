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
@Desc       :   Sec 4.4 自定义层
@小结：
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_figures, show_title


# ----------------------------------------------------------------------
class CenteredLayer(nn.Block):
    """
    这个层不含模型参数，只对数据的均值归 0
    """

    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
        pass

    def forward(self, x):
        return x - x.mean()

    pass


class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))
        pass

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        # print(self.weight.data())
        # print(self.bias.data())
        # print(linear)
        return nd.relu(linear)

    pass


def main():
    # no_params_custom_layer()

    show_title("4.4.2 包含模型参数的自定义层")
    # construct_model_params()

    dense = MyDense(units=3, in_units=5)
    show_subtitle("模型参数")
    print(dense.params)
    dense.initialize()
    show_subtitle("模型权重矩阵")
    print(dense.weight.data())
    show_subtitle("模型偏差")
    print(dense.bias.data())
    show_subtitle("模型输出结果")
    print(dense(nd.array([1, 2, 3, 4, 5])))
    # print(dense(nd.random.uniform(shape=(2, 5))))

    show_subtitle("构造模型")
    net = nn.Sequential()
    net.add(MyDense(units=8, in_units=64),
            MyDense(units=1, in_units=8))
    net.initialize()
    print(net(nd.random.uniform(shape=(4, 64))))
    # ToDo: 为什么计算输出的值都是 0？
    pass


def construct_model_params():
    show_subtitle("创建模型参数")
    params = gluon.ParameterDict()
    params.get('param2', shape=(2, 3))
    print(params)


def no_params_custom_layer():
    show_title("4.4.1 不含模型参数的自定义层")
    show_subtitle("自定义层输出效果")
    layer = CenteredLayer()
    print(layer(nd.array([1, 2, 3, 4, 5])))
    show_subtitle("更复杂模型的输出效果")
    net = nn.Sequential()
    net.add(nn.Dense(16),
            CenteredLayer())
    net.initialize()
    print("模型输出的均值为：", net(nd.random.uniform(shape=(4, 8))).mean().asscalar())
    print("模型的参数为：", net.collect_params())


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
