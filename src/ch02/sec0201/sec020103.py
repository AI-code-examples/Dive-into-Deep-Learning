"""
=================================================
@path   : Dive-into-Deep-Learning -> sec020103.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-23 15:37
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   2.1.数据操作→2.1.3.广播机制
==================================================
"""
import torch
from datetime import datetime
from tools import beep_end, show_subtitle


def main(name):
    print(f'Hi, {name}', datetime.now())
    show_subtitle("broadcasting")
    A = torch.arange(3).reshape((3, 1))
    B = torch.arange(2).reshape((1, 2))
    print("A=", A)
    print("B=", B)
    print("A+B=", A + B)
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
