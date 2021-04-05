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
    attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
    attention.initialize()

    batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, np.array([3, 2])
    X = np.ones((batch_size, num_queries, num_hiddens))
    Y = np.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)
    pass


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of 'transpose_qkv'"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Block):
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # 'queries' 的形状：('batch_size', 查询或者“键－值”对的个数, 'num_hiddens')
        # 'valid_lens' 的形状：('batch_size',) 或者 ('batch_size', 查询的个数)
        # 变换后，输出的 'queries', 'keys', 'values' 的形状：
        # ('batch_size'*'num_heads', 查询或者“键－值”对的个数, 'num_hiddens'/'num_heads')
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在 axis=0，拷贝第一项（标题或者失量）'num_heads' 次；然后拷贝下一项；等等
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # 'output' 的形状：('batch_size'*'num_heads', 查询的个数, 'num_hiddens'/'num_heads')
        output = self.attention(queries, keys, values, valid_lens)
        # 'output_concat' 的形状：('batch_size', 查询的个数, 'num_hiddens')
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
