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
from mxnet.gluon import nn, rnn
from tools import beep_end, show_subtitle, show_title, show_figures

npx.set_np()


# ----------------------------------------------------------------------
def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    init_lstm_state = lambda batch_size, num_hiddens, device: (
        np.zeros((batch_size, num_hiddens), ctx=device),
        np.zeros((batch_size, num_hiddens), ctx=device))

    # 9.2.2.3 Training and Prediction
    vocab_size, num_hiddens = len(vocab), 256
    num_epochs, lr = 500, 1
    device = d2l.try_gpu()
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    lstm_layer = rnn.LSTM(num_hiddens)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    lstm_layer2 = rnn.LSTM(num_hiddens, num_layers=2)
    model2 = d2l.RNNModel(lstm_layer2, len(vocab))
    d2l.train_ch8(model2, train_iter, vocab, lr, num_epochs * 2, device)
    pass


def get_lstm_params(vocab_size, num_hiddens, device):
    """Initializing Model Parameters."""
    num_inputs = num_outputs = vocab_size
    normal = lambda shape: np.random.normal(scale=0.01, size=shape, ctx=device)
    zeros = lambda num_cells: np.zeros(num_cells, ctx=device)
    three = lambda: (normal((num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)), zeros(num_hiddens))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = zeros(num_outputs)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def lstm(inputs, state, params):
    """Defining the Model."""
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = npx.sigmoid(np.dot(X, W_xi) + np.dot(H, W_hi) + b_i)
        F = npx.sigmoid(np.dot(X, W_xf) + np.dot(H, W_hf) + b_f)
        O = npx.sigmoid(np.dot(X, W_xo) + np.dot(H, W_ho) + b_o)
        C_tilda = np.tanh(np.dot(X, W_xc) + np.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * np.tanh(C)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H, C)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
