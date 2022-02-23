"""
=================================================
@path   : Dive-into-Deep-Learning -> sec020102.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-23 15:24
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   2.1.数据操作→2.1.2.运算符
==================================================
"""
import torch

from datetime import datetime


def element_add(X, Y):
    print("X+Y=", X + Y)
    pass


def element_multiply(X, Y):
    print("X*Y=", X * Y)
    pass


def element_divide(X, Y):
    print("X/Y=", X / Y)
    pass


def element_exp(Y):
    print("Y.exp()=", Y.exp())
    pass


def array_concat(X, Y):
    Z0 = torch.cat((X, Y), dim=0)
    Z1 = torch.cat((X, Y), dim=1)
    print("Z0=", Z0)
    print("Z1=", Z1)
    pass


def array_equal(X, Y):
    print("X=", X)
    print("Y=", Y)
    print("X==Y", X == Y)
    pass


def array_sum(X):
    print("X.sum()=", X.sum())
    pass


def array_normal(X):
    """返回范数"""
    print("X.norm()=", X.norm())
    pass


def main(name):
    print(f'Hi, {name}', datetime.now())
    X = torch.tensor([1.0, 2, 4, 8])
    Y = torch.tensor([2, 2, 2, 2])
    element_add(X, Y)
    element_multiply(X, Y)
    element_divide(X, Y)
    element_exp(Y)
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    array_concat(X, Y)
    array_equal(X, Y)
    array_sum(X)
    array_normal(X)
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
