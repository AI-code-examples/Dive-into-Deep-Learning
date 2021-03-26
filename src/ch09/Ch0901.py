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

    # Training and Prediction
    vocab_size, num_hiddens = len(vocab), 256
    device = d2l.try_gpu()
    num_epochs, lr = 500, 1
    init_gru_state = lambda batch_size, num_hiddens, device: (np.zeros(shape=(batch_size, num_hiddens), ctx=device),)
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    gru_layer = rnn.GRU(num_hiddens)
    model = d2l.RNNModel(gru_layer, len(vocab))
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    gru_layer=rnn.GRU(num_hiddens,2)
    model2 = d2l.RNNModel(gru_layer, len(vocab))
    d2l.train_ch8(model2, train_iter, vocab, lr, num_epochs, device)
    pass


def get_params(vocab_size, num_hiddens, device):
    """Initializing Model Parameters"""
    num_inputs = num_outputs = vocab_size
    normal = lambda shape: np.random.normal(scale=0.01, size=shape, ctx=device)
    zeros = lambda num_cell: np.zeros(num_cell, ctx=device)
    three = lambda: (normal((num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)), zeros(num_hiddens))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate Parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq, b_q = normal((num_hiddens, num_outputs)), zeros(num_outputs)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def gru(inputs, state, params):
    """Defining the Model"""
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = npx.sigmoid(np.dot(X, W_xz) + np.dot(H, W_hz) + b_z)
        R = npx.sigmoid(np.dot(X, W_xr) + np.dot(H, W_hr) + b_r)
        H_tilda = np.tanh(np.dot(X, W_xh) + np.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)


# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
# 运行结束的提醒
beep_end()
show_figures()
