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
    # Load data
    batch_size, num_steps, device = 32, 35, d2l.try_gpu()
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # Definie the bidirectional LSTM model by setting `bidirectional=True`
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    lstm_layer = rnn.LSTM(num_hiddens, num_layers, bidirectional=True)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    # Train the model
    num_epochs, lr = 500, 1
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
