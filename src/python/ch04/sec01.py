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
@Desc       :   Sec 4.1 基于 Block类的模型构造
@小结：
1.  可以通过继承 Block 类来构造模型
2.  Sequential 类继承自 Block 类
3.  虽然使用 Sequential 类构造模型会更简单，但是通过继承 Block 类构造模型会更灵活
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_figures


class MLP(nn.Block):
    def __init__(self, **kwargs):
        # 调用父类完成初始化
        # 参数（Ref：4.2）
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)
        pass

    def forward(self, x):
        return self.output(self.hidden(x))


class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)
        pass

    def add(self, block):
        # block 是一个 Block 子类实例
        # _children 是成员变量，类型为 OrderedDict
        # 当 MySequential 实例调用 initialize() 时，系统会对 _children 中的所有成员初始化
        self._children[block.name] = block
        pass

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
            pass
        return x


class FancyMLP(nn.Block):
    """
    模型中使用了常数权重 rand_weight 与 数据进行矩阵乘
    注：这个常数权重不是模型参数
    模型中还重复使用了相同的 Dense 层
    """

    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant('rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')
        pass

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        x = self.dense(x)
        while x.norm().asscalar() > 1:
            x /= 2
            pass
        if x.norm().asscalar() < 0.8:
            x *= 10
            pass
        return x.sum()


class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')
        pass

    def forward(self, x):
        return self.dense(self.net(x))


# ----------------------------------------------------------------------
def main():
    X = nd.random.uniform(shape=(2, 20))
    # net = MLP_Block()
    # net = Sequential_Block(X)
    # net = FancyMLP_Block()
    net = NestMLP_Block()
    print(net(X))
    pass


def NestMLP_Block():
    net = nn.Sequential()
    net.add(NestMLP(), nn.Dense(20), FancyMLP())
    net.initialize()
    return net


def FancyMLP_Block():
    net = FancyMLP()
    net.initialize()
    return net


def Sequential_Block():
    show_subtitle("4.1.2 继承 Block 类构造 Sequential 类风格的子类")
    net = MySequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    net.initialize()
    return net


def MLP_Block():
    show_subtitle("4.1.1 继承 Block 类来构造模型")
    net = MLP()
    net.initialize()
    return net


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
