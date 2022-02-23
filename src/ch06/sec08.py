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
@Desc       :   Sec 6.8 长短期记忆（LSTM）
@小结：
1.  长短期记忆的隐藏层输出=隐藏状态+记忆细胞，只有隐藏状态会传递到输出层
2.  长短期记忆的输入门、遗忘门和输出门可以控制信息的流动
3.  长短期记忆可以应对循环神经网络中的梯度衰减问题，并且能够更好地捕捉时间序列中时间步距离较大的依赖关系
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

        def _three():
            return (_one((num_inputs, num_hiddens)),
                    _one((num_hiddens, num_hiddens)),
                    nd.zeros(num_hiddens, ctx=ctx))

        W_xi, W_hi, b_i = _three()
        W_xf, W_hf, b_f = _three()
        W_xo, W_ho, b_o = _three()
        W_xc, W_hc, b_c = _three()
        W_hq = _one((num_hiddens, num_outputs))
        b_q = nd.zeros(num_outputs, ctx=ctx)
        params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
        for param in params:
            param.attach_grad()
            pass
        return params

    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ["分开", "不分开"]
    d2l.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx, False,
                              num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
    lstm_layer = rnn.LSTM(num_hiddens)
    model = d2l.RNNModel(lstm_layer, vocab_size)
    d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx, num_epochs, num_steps, lr,
                                    clipping_theta, batch_size, pred_period, pred_len, prefixes)
    pass


def init_lstm_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),
            nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx))


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
        pass
    return outputs, (H, C)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
