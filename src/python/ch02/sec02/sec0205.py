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
@Desc       :   Sec 2.2.5 运算的内存开销
@理解：
"""
from mxnet import nd, ndarray
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def memory_change(X, Y):
    show_subtitle("memory change")
    before = id(Y)
    Y = Y + X
    after = id(Y)
    print("before=\t", before)
    print("after=\t", after)
    print("(after == before) =", after == before)
    pass


def memory_assign(X, Y):
    """
    X+Y 需要占用临时内存，再拷贝到 Z
    """
    show_subtitle("memory assign")
    Z = Y.zeros_like()
    before = id(Z)
    Z[:] = X + Y
    after = id(Z)
    print("before=\t", before)
    print("after=\t", after)
    print("(after == before) =", after == before)
    pass


def memory_fast_assign(X, Y):
    """
    X+Y 不需要占用临时内存，直接输出到 Z
    """
    show_subtitle("memory fast assign")
    Z = Y.zeros_like()
    before = id(Z)
    nd.elemwise_add(X, Y, out=Z)
    after = id(Z)
    print("before=\t", before)
    print("after=\t", after)
    print("(after == before) =", after == before)
    pass


def memory_direct_assign(X, Y):
    """
    直接使用原始变量的内存空间
    """
    show_subtitle("memory direct assign")
    before = id(X)
    X += Y
    after = id(X)
    print("before=\t", before)
    print("after=\t", after)
    print("(after == before) =", after == before)
    pass


def main():
    X = (nd.arange(12)).reshape((3, 4))
    Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    memory_change(X, Y)
    memory_assign(X, Y)
    memory_fast_assign(X, Y)
    memory_direct_assign(X, Y)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
