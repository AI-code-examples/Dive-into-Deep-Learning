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
@Desc       :   Sec 3.1 线性回归的基本描述
@理解：
"""
from time import time

from mxnet import nd
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def main():
    vector_calculator()
    pass


def vector_calculator():
    a = nd.ones(shape=1000)
    b = nd.ones(shape=1000)
    show_subtitle("失量的元素加")
    start = time()
    c = nd.zeros(shape=1000)
    for i in range(1000):
        c[i] = a[i] + b[i]
    end = time()
    print("time consume=", end - start)

    show_subtitle("失量加")
    start = time()
    c = a + b
    end = time()
    print("time consume=", end - start)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
