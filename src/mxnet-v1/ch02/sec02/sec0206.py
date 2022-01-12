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
@Desc       :   Sec
@理解：
"""
import numpy as np
from mxnet import nd, ndarray
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def ndarray_numpy():
    """
    新版本的 mxnet==1.7.0 已经直接使用 numpy 了
    """
    P = np.ones((2, 3))
    D = nd.array(P)
    show_subtitle("ndarray & numpy : P")
    print(P)
    print("id(P)=", id(P))
    show_subtitle("ndarray & numpy : D")
    print(D)
    print("id(D)=", id(D))
    show_subtitle("ndarray & numpy : D.asnumpy()")
    print(D.asnumpy())
    print("id(D.asnumpy())=", id(D.asnumpy()))
    pass


def main():
    ndarray_numpy()
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
