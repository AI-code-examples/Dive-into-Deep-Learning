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
@Desc       :   Sec 10.1 词嵌入（Word2Vec）
@小结：
1.  词向量是用来表示词的向量。把词映射为实数域向量的技术也叫词嵌入
2.  Word2Vec 包含跳字模型和连续词袋模型。
    -   跳字模型假设基于中心词来生成背景词
    -   连续词袋模型假设基于背景词来生成中心词
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
