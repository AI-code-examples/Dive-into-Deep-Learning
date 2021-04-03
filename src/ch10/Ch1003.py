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
import math

from d2l import mxnet as d2l
from mxnet import nd, np, npx, autograd
from mxnet.gluon import nn
from tools import beep_end, show_subtitle, show_title, show_figures

npx.set_np()


# ----------------------------------------------------------------------
def main():
    data = np.random.uniform(size=(2, 2, 4))
    print(data)
    print(masked_softmax(data, np.array([2, 3])))

    # 10.3.2. Additive Attention
    keys = np.ones((2, 10, 2))
    # The two value matrices in the `values` minibatch are identical
    values = np.arange(40).reshape(1, 10, 4).repeat(2, axis=0)
    valid_lens = np.array([2, 6])

    queries = np.random.normal(0, 1, (2, 1, 20))
    attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
    attention.initialize()
    print(attention(queries, keys, values, valid_lens))
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')

    queries = np.random.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.initialize()
    attention(queries, keys, values, valid_lens)
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
    pass


class DotProductAttention(nn.Block):
    """Scaled dot product attention."""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # `queries` shape: (`batch_size`, no. of queries, `d`)
        # 'keys` shape: (`batch_size`, no. of key-value pairs, `d`)
        # `values` shape: (`batch_size`, no. of key-value pairs, `d`)
        # `valid_lens` shape: (`batch_size`,) or (`batch_size`, no. of queries)
        d = queries.shape[-1]
        # `transpose_b=True`: swap the last two dimensions of `keys`
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)


class AdditiveAttention(nn.Block):
    """Additive attention."""

    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # Use `flatten=False` to only transform the last axis
        # so that the shapes for the other axes kept the same
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion
        # `queries` shape: (`batch_size`, no. of queries, `num_hiddens`)
        # `keys` shape: (`batch_size`, no. of key-value pairs, `num_hiddens`)
        # Sum them up with broadcasting
        features = np.expand_dims(queries, axis=2) + np.expand_dims(keys, axis=1)
        features = np.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last one-dimensional entry from the shape.
        # `scores` shape: (`batch_size`, no. of queries, no. of key-value pairs)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # `values` shape: (`batch_size`, no. of key-value pairs, value dimension)
        return npx.batch_dot(self.dropout(self.attention_weights), values)


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, use_sequence_length=True, value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
