"""
=================================================
@path   : Dive-into-Deep-Learning -> sec020106.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-23 15:51
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   2.1.数据操作→2.1.6.Python对象转换
==================================================
"""
import torch
import numpy as np
from datetime import datetime
from tools import beep_end, show_subtitle


def main(name):
    print(f'Hi, {name}', datetime.now())
    P = np.ones((2, 3))
    D = torch.tensor(P)
    show_subtitle("tensor & numpy : P")
    print(P)
    print("id(P)=", id(P))
    show_subtitle("tensor & numpy : D")
    print(D)
    print("id(D)=", id(D))
    show_subtitle("tensor & numpy : D.asnumpy()")
    print(np.array(D))
    print("id(D.asnumpy())=", id(np.array(D)))
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    beep_end()
