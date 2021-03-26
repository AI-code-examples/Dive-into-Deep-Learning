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

    # 8.6.1 Defining the Model
    num_hiddens = 256
    rnn_layer = rnn.RNN(num_hiddens)
    rnn_layer.initialize()

    state = rnn_layer.begin_state(batch_size=batch_size)
    len(state), state[0].shape

    X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)
    Y.shape, len(state_new), state_new[0].shape

    # 8.6.2 Training and Predicting
    device = d2l.try_gpu()
    net = RNNModel(rnn_layer, len(vocab))
    net.initialize(force_reinit=True, ctx=device)
    d2l.predict_ch8('time traveller ', 10, net, vocab, device)

    num_epochs, lr = 500, 1
    d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
    d2l.predict_ch8('time traveller', 10, net, vocab, device)

    pass


class RNNModel(nn.Block):
    """The RNN model."""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
