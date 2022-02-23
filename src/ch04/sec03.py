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
@Desc       :   Sec 4.3 模型参数的延后初始化
@小结：
1.  延后初始化：系统在得到足够信息后才对模型参数进行初始化
2.  延后初始化的优点：模型构造更加简单
3.  强制描述模型参数的形状可以对模型立即进行初始化
"""

from mxnet import init, nd
from mxnet.gluon import nn

from tools import beep_end, show_subtitle, show_figures, show_title


# ----------------------------------------------------------------------
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 通过 print() 函数观察初始化的时间

    pass


def instant_init_parameters():
    """
    指定第一个隐藏层的参数形状就可以立即初始化
    """
    show_title("4.3.2 模型参数立即初始化")
    net = nn.Sequential()
    net.add(nn.Dense(256, in_units=20, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(10, in_units=256))
    net.initialize(init=MyInit())
    # Note: 观察结果会发现模型在执行前向计算前对层与层之间的关系是不了解的，因此无法进行参数形状推断
    pass


def main():
    delay_init_parameters()
    instant_init_parameters()
    pass


def delay_init_parameters():
    """
    延后初始化是因为模型需要根据输入数据的形状来推断参数的形状
    """
    show_title("4.3.1 模型参数延后初始化")
    X = nd.random.uniform(shape=(2, 20))
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(10))
    show_subtitle("调用初始化函数时未进行模型初始化")
    net.initialize(init=MyInit())
    show_subtitle("执行前向计算时才进行模型初始化")
    net(X)
    print("模型参数：", net[0].weight.data()[0])
    show_subtitle("再执行前向计算时不再进行模型初始化")
    print("模型参数：", net[0].weight.data()[0])


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
