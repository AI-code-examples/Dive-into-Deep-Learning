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
@Desc       :   Sec 2.2.4 索引
@理解：
"""
from mxnet import nd
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def main():
    X = (nd.arange(12)).reshape((3, 4))
    show_subtitle("index & slice")
    print("X[1:3]=", X[1:3])
    show_subtitle("X[1,2]=9")
    X[1, 2] = 9
    print("X=", X)
    show_subtitle("X[1:2,:]=12")
    X[1:2, :] = 12
    print("X=", X)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
