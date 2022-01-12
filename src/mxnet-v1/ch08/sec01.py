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
@Desc       :   Sec 8.1 命令式编程+符号式编程
@小结：
1.  命令式编程（imperative programming)与符号式编程(symbolic programming)各有优劣，MXNet可以混合式编程
    -   命令式编程更加方便：代码编写和调试都更容易
    -   符号式编程更加高效也更容易移植：编译的时候系统容易做更多优化；编译后程序变成一个与Python无关的格式，从而可以使程序在非 Python环境下运行，避开了Python解释器的性能问题
2.  通过 HybridNet 类和 HybridBlock 类构建的模型可以调用 hybridize 函数将命令式程序转成符号式程序，从而提高计算性能
"""
import time

import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # imperative_programming()

    # symbolic_programming()

    # hybrid_sequential_construct()

    class HybridNet(nn.HybridBlock):
        def __init__(self, **kwargs):
            super(HybridNet, self).__init__(**kwargs)
            self.hidden = nn.Dense(10)
            self.output = nn.Dense(2)
            pass

        def hybrid_forward(self, F, x):
            # x.asnumpy() # 没有实现的函数就无法使用
            # ToDo：如果加入 if 或 for 语句会如何？
            print("F: ", F)
            print("x: ", x)
            x = F.relu(self.hidden(x))
            print("hidden: ", x)
            return self.output(x)

        pass

    net = HybridNet()
    net.initialize()
    x = nd.random.normal(shape=(1, 4))
    print("Output: ", net(x))
    net.hybridize()
    show_subtitle("net.hybridize()")
    print("net(x): ", net(x))
    # 第二次访问时，符号式程序已经编译完成，就不会再访问 Python 的代码，而在 C++ 后端执行符号式程序
    # 提升了性能，失去了程序的灵活性
    # 原地（in-place)操作：a+=b,a[:]=a+b 不被符号式编程支持，需要改成 a=a+b
    show_subtitle("net.hybridize()")
    print("net(x): ", net(x))
    pass


def hybrid_sequential_construct():
    # 网络工厂
    def get_net():
        net = nn.HybridSequential()
        net.add(nn.Dense(256, activation='relu'),
                nn.Dense(128, activation='relu'),
                nn.Dense(2))
        net.initialize()
        return net

    # 混合编程
    x = nd.random.normal(shape=(1, 512))
    net = get_net()
    print("net(x)=", net(x))
    net.hybridize()
    print("net.hybridize(),net(x)=", net(x))

    # 1. 性能提升对比
    def benchmark(net, x):
        start = time.time()
        for i in range(1000):
            _ = net(x)
            pass
        nd.waitall()  # 等待所有计算完成
        return time.time() - start

    net = get_net()
    print("before hybridizing: %.4f sec" % (benchmark(net, x)))
    net.hybridize()
    print("after hybridizing: %.4f sec" % (benchmark(net, x)))
    # 2. 获取符号式程序
    net.export('my_mlp')


def symbolic_programming():
    def add_str():
        return '''
def add(a,b):
    return a+b

print(add(1,2))
    '''

    def fancy_func_str():
        # ToDo：与书中情况不符，不允许跨域访问其他函数
        return '''
def fancy_func(a,b,c,d):
    def add(a,b):
        return a+b
    e=add(a,b)
    f=add(c,d)
    g=add(e,f)
    return g
print(fancy_func(1,2,3,4))
    '''

    def evoke_str():
        return add_str() + fancy_func_str()

    prog = evoke_str()
    print(prog)
    y = compile(prog, '', 'exec')
    exec(y)
    pass


def imperative_programming():
    def add(a, b):
        return a + b

    def fancy_func(a, b, c, d):
        e = add(a, b)
        f = add(c, d)
        g = add(e, f)
        return g

    print(fancy_func(1, 2, 3, 4))
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
