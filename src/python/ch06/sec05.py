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
@Desc       :   Sec 6.5 循环神经网络的 Gluon 实现
@小结：
1.  Gluon 的 rnn 模块提供了循环神经网络的实现
2.  Gluon 的 rnn.RNN 实例在前向计算后会分别返回输出和隐藏状态。这个前向计算并不涉及输出层的计算
"""
import math
import time

import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn

from data import load_data_jay_lyrics
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

    num_hiddens = 256
    rnn_layer = rnn.RNN(num_hiddens)
    rnn_layer.initialize()
    batch_size = 2
    state = rnn_layer.begin_state(batch_size=batch_size)
    print(state)
    print(state[0].shape)

    num_steps = 35
    X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, len(state_new), state_new[0].shape)

    ctx = d2l.try_gpu()
    model = RNNModel(rnn_layer, vocab_size)
    model.initialize(ctx=ctx, force_reinit=True)
    print(predict_rnn_gluon("分开", 10, model, vocab_size, ctx, idx_to_char, char_to_idx))

    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ["分开", "不分开"]
    train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx, num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
    pass


class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)
        pass

    def forward(self, inputs, state):
        # 将输入转置成 (num_steps, batch_size) 后获取 one-hot 向量表示
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层
        # 首先，将 Y 的形状变成 (num_steps*batch_size,num_hiddens)
        # 然后，输出形状为 (num_steps*batch_size, vocab_size)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

    pass


def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char, char_to_idx):
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
            pass
        pass
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_gluon(model: RNNModel, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(init=init.Normal(0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0, 'wd': 0})
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
                pass
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
                pass
            l.backward()
            params = [p.data() for p in model.collect_params().values()]
            d2l.grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)
            l_sum += l.asscalar() * y.size
            n += y.size
            pass
        if (epoch + 1) % pred_period == 0:
            print("epoch %d, perplexity %f, time %.2f sec" % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(" -", predict_rnn_gluon(prefix, pred_len, model, vocab_size, ctx, idx_to_char, char_to_idx))
                pass
            pass
        pass
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
