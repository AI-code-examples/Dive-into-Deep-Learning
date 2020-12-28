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
@Desc       :   Sec 2.2.3 广播机制
@理解：
"""
from mxnet import nd
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def main():
    show_subtitle("broadcasting")
    A = nd.arange(3).reshape((3, 1))
    B = nd.arange(2).reshape((1, 2))
    print("A=", A)
    print("B=", B)
    print("A+B=", A + B)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
