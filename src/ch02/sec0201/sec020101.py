# -*- encoding: utf-8 -*-
"""
=================================================
@path   : Dive-into-Deep-Learning -> sec020101.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-23 15:22
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   2.1.数据操作→2.1.1.入门
==================================================
"""
import torch
from tools import beep_end
from datetime import datetime

# ----------------------------------------------------------------------


def create_data():
    x = torch.arange(12)
    print("x=", x)
    data_size(x)
    return x


def data_size(x):
    print("x.shape=", x.shape)
    print("x.size()=", x.size())


def reshape_data(X):
    x = X.reshape(3, 4)
    print("x.reshape=", x)
    data_size(x)
    return x


def zero_data():
    x = torch.zeros((2, 3, 4))
    print("zero x=", x)
    data_size(x)
    return x


def one_data():
    x = torch.ones((3, 4))
    print("one x=", x)
    data_size(x)
    return x


def init_data():
    y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print("init y=", y)
    data_size(y)
    return y


def random_data():
    x = torch.randn(3, 4)
    print("random x=", x)
    data_size(x)
    pass


def main(name):
    print(f'Hi, {name}', datetime.now())
    print('-' * 30)
    X = create_data()
    print('-' * 30)
    reshape_data(X)

    print('-' * 30)
    zero_data()
    print('-' * 30)
    one_data()

    print('-' * 30)
    init_data()
    print('-' * 30)
    random_data()


if __name__ == '__main__':
    __author__ = 'zYx.Tom'
    main(__author__)
    beep_end()
