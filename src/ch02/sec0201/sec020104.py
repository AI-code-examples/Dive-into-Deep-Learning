"""
=================================================
@path   : Dive-into-Deep-Learning -> sec020104.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-23 15:41
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   2.1.数据操作→2.1.4.索引与切片
==================================================
"""
import torch
from datetime import datetime
from tools import beep_end, show_subtitle


def main(name):
    print(f'Hi, {name}', datetime.now())
    X = (torch.arange(12)).reshape((3, 4))
    show_subtitle("index & slice")
    print("X[1:3]=", X[1:3])
    show_subtitle("X[1,2]=9")
    X[1, 2] = 9
    print("X=", X)
    show_subtitle("X[1:2,:]=12")
    X[1:2, :] = 12
    print("X=", X)
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)