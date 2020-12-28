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
@Desc       :   Sec 2.2.2 运算
@理解：
"""
from mxnet import nd
from tools import beep_end


# ----------------------------------------------------------------------
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
    Z0 = nd.concat(X, Y, dim=0)
    Z1 = nd.concat(X, Y, dim=1)
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
    print("X.norm()=", X.norm())
    print("X.norm().asscalar()=", X.norm().asscalar())
    pass


def main():
    X = (nd.arange(12)).reshape((3, 4))
    Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    element_add(X, Y)
    element_multiply(X, Y)
    element_divide(X, Y)
    element_exp(Y)
    array_concat(X, Y)
    array_equal(X, Y)
    array_sum(X)
    array_normal(X)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
