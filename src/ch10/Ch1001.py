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
@小结：
"""
import random

from d2l import mxnet as d2l
from mxnet import nd, np, npx, autograd
from mxnet.gluon import nn
from tools import beep_end, show_subtitle, show_title, show_figures

npx.set_np()


# ----------------------------------------------------------------------
def main():
    attention_weights = np.eye(10).reshape((1, 1, 10, 10))
    attention_weights = np.random.normal(size=(10, 10)).reshape((1, 1, 10, 10))
    show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
    pass


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.asnumpy(), cmap=cmap)
            if i == num_rows - 1: ax.set_xlabel(xlabel)
            if j == 0: ax.set_ylabel(ylabel)
            if titles: ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
