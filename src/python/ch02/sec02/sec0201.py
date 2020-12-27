# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   Dive-into-Deep-Learning
@File       :   sec0201.py
@Version    :   v0.1
@Time       :   2020-12-27 上午11:30
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   2.2.1 创建 NDArray
@理解
"""
from mxnet import nd
from tools import beep_end


# ----------------------------------------------------------------------


def create_data():
    x = nd.arange(12)
    print("x=", x)
    print("x.shape=", x.shape)
    print("x.size=", x.size)
    return x


def reshape_data(X):
    x = X.reshape(3, 4)
    print("x.reshape=", x)
    return x


def zero_data():
    x = nd.zeros((2, 3, 4))
    print("zero x=", x)
    return x


def one_data():
    x = nd.ones((3, 4))
    print("one x=", x)
    return x


def init_data():
    y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print("init y=", y)
    return y


def random_data():
    x = nd.random.normal(0, 1, shape=(3, 4))
    print("random x=", x)
    pass


def main():
    X = create_data()
    reshape_data(X)

    zero_data()
    one_data()

    init_data()
    random_data()


if __name__ == '__main__':
    main()
    beep_end()
