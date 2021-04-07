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
    num_hiddens, num_heads = 100, 5
    attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
    attention.initialize()

    batch_size, num_queries, valid_lens = 2, 4, np.array([3, 2])
    X = np.ones((batch_size, num_queries, num_hiddens))
    attention_weights = attention(X, X, X, valid_lens)
    print(attention_weights.shape)

    # 10.6.3. Positional Encoding
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.initialize()
    X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)', figsize=(6, 2.5), legend=['Col %d' % d for d in np.arange(6, 10)])

    # 10.6.3.1. Absolute Positional Information
    for i in range(8):
        print(f'{i} in binary is {i:>03b}')

    P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
    d2l.show_heatmaps(P, xlabel='Column (encoding dimension)', ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')

    # 10.6.3.2. Relative Positional Information

    pass


class PositionalEncoding(nn.Block):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len).reshape(-1, 1) / np.power(10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
