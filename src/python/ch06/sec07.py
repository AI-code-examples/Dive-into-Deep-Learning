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
@Desc       :   Sec 6.7 门控循环单元（GRU）
@小结：
1.  门控循环神经网络可以更好地捕捉时间序列中时间步距离较大的依赖关系
2.  门控循环单元引入了门的概念，从而修改了循环神经网络中隐藏状态的计算方式。
    包括：重置门、更新门、候选隐藏状态和隐藏状态
3.  重置门有助于捕捉时间序列里短期的依赖关系
4.  更新门有助于捕捉时间序列里长期的依赖关系
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn
from tools import beep_end, show_subtitle, show_title, show_figures
from data import load_data_jay_lyrics


# ----------------------------------------------------------------------
def main():
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    ctx = d2l.try_gpu()

    def get_params():
        def _one(shape):
            return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

        pass

        def _three():
            return (_one((num_inputs, num_hiddens)),
                    _one((num_hiddens, num_hiddens)),
                    nd.zeros(num_hiddens, ctx=ctx))

        W_xz, W_hz, b_z = _three()
        W_xr, W_hr, b_r = _three()
        W_xh, W_hh, b_h = _three()
        W_hq = _one((num_hiddens, num_outputs))
        b_q = nd.zeros(num_outputs, ctx=ctx)
        params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.attach_grad()
            pass
        return params

    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ["分开", "不分开"]
    print(d2l.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx, False, num_epochs,
                                    num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes))

    gru_layer = rnn.GRU(num_hiddens)
    model = d2l.RNNModel(gru_layer, vocab_size)
    print(d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx, num_epochs, num_steps, lr,
                                          clipping_theta, batch_size, pred_period, pred_len, prefixes))

    pass


def init_gru_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + b_z)
        R = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + b_r)
        H_tilda = nd.tanh(nd.dot(X, W_xh) + nd.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
        pass
    return outputs, (H,)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
