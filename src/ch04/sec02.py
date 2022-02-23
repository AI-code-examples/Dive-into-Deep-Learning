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
@Desc       :   Sec 4.2 模型参数的初始化、访问和共享
@小结：
1.  有多种方法可以访问模型的参数
2.  有多种方法可以初始化模型的参数
3.  可以自定义初始化模型参数的方法
4.  共享模型参数的方法？
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_figures, show_title


# ----------------------------------------------------------------------
def main():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    net.initialize()  # 默认初始化方法
    X = nd.random.uniform(shape=(2, 20))
    net(X)  # 前向计算

    # get_model_parameters(net)
    # init_model_parameters(net)
    share_model_parameters(X)
    pass


def share_model_parameters(X):
    show_title("4.2.4 共享模型参数")
    net = nn.Sequential()
    hidden1 = nn.Dense(8, activation='relu')
    shared = nn.Dense(8, activation='relu')
    hidden3 = nn.Dense(8, activation='relu', params=shared.params)
    output = nn.Dense(10)
    net.add(hidden1, shared, hidden3, shared, output)
    net.initialize()
    net(X)
    # ToDo：共享参数与共享网络的区别？
    show_subtitle("第二隐藏层（共享参数）的参数")
    print(net[1].weight.data()[0])
    show_subtitle("第三隐藏层的参数")
    print(net[2].weight.data()[0])
    show_subtitle("第四隐藏层（共享网络）的参数")
    print(net[3].weight.data()[0])


class MyInit(init.Initializer):
    """
    权重一半概率初始化为0；一半概率初始化为[-10,-5]和[5,10]两个区间里面均匀分布的随机数
    """

    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5


def init_model_parameters(net):
    show_title("4.2.2 初始化模型参数")

    show_subtitle("未强制初始化得到的第一层的权重参数的值")
    print(net[0].weight.data()[0])

    # force_reinit 是强制初始化，用于已经初始化过的模型的再次初始化

    show_subtitle("强制初始化得到的第一层的权重参数的值")
    net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
    print(net[0].weight.data()[0])

    show_subtitle("强制常数初始化得到的第一层的权重参数的值")
    net.initialize(init=init.Constant(1), force_reinit=True)
    print(net[0].weight.data()[0])

    show_subtitle("对第一层使用 Xavier 强制初始化权重参数的值")
    net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
    print(net[0].weight.data()[0])

    show_subtitle("自定义初始化得到的第一层的权重参数的值")
    net.initialize(init=MyInit(), force_reinit=True)
    print(net[0].weight.data()[0])

    show_subtitle("set_data() 修改模型参数得到的第一层的权重参数的值")
    net[0].weight.set_data(net[0].weight.data() + 1)
    print(net[0].weight.data()[0])


def get_model_parameters(net):
    show_title("4.2.1 访问模型参数")
    show_subtitle("第一层的参数")
    print(net[0].params)
    show_subtitle("第一层的参数的类型")
    print(type(net[0].params))
    show_subtitle("第一层的权重参数")
    print(net[0].params['dense0_weight'])
    show_subtitle("第一层的权重参数")
    print(net[0].weight)
    show_subtitle("第一层的权重参数的值")
    print(net[0].weight.data())
    show_subtitle("第一层的权重参数的梯度")
    print(net[0].weight.grad())
    show_subtitle("第二层的偏差的值")
    print(net[1].bias.data())
    show_subtitle("所有模型参数")
    print(net.collect_params())
    show_subtitle("基于正则表达式匹配的模型参数")
    print(net.collect_params('.*weight'))


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
