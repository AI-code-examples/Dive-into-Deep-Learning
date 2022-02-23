"""
=================================================
@path   : Dive-into-Deep-Learning -> sec0301.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-23 15:22
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   3.1.线性回归的基本概念
==================================================
"""
import math

import numpy as np
import torch
from d2l import torch as d2l

from tools import beep_end
# ----------------------------------------------------------------------
from tools import show_figures
from tools import show_subtitle
from tools import Timer


def main():
    vector_calculator()

    x = np.arange(-7, 7, 0.01)
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params],
             xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5),
             legend=[f'mean{mu},std{sigma}' for mu, sigma in params])

    pass


def vector_calculator():
    a = torch.randn(100000)
    b = torch.randn(100000)
    c = torch.zeros(100000)
    timer = Timer()
    show_subtitle("失量的元素加")
    timer.start()
    for i in range(100000):
        c[i] = a[i] + b[i]
    print(f"time consume={timer.stop():.5f}")

    show_subtitle("失量加")
    timer.start()
    c = a + b
    print(f"time consume={timer.stop():.5f}")


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
    show_figures()
