"""
=================================================
@path   : Dive-into-Deep-Learning -> sec020105.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-23 15:44
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   2.1.数据操作→2.1.5.节省内存
==================================================
"""
import torch
from datetime import datetime
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
    Z = torch.zeros_like(Y)
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
    Z = torch.zeros_like(Y)
    before = id(Z)
    Z[:] = X + Y
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


def main(name):
    print(f'Hi, {name}', datetime.now())
    X = (torch.arange(12)).reshape((3, 4))
    Y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    memory_change(X, Y)
    memory_assign(X, Y)
    memory_fast_assign(X, Y)
    memory_direct_assign(X, Y)
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    beep_end()
