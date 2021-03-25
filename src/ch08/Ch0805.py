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
import matplotlib.pyplot as plt
import random, math

from d2l import mxnet as d2l
from mxnet import nd, np, npx, autograd, gluon, init
from mxnet.gluon import nn
from tools import beep_end, show_subtitle, show_title, show_figures

npx.set_np()


# ----------------------------------------------------------------------
def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    for i, text in enumerate(train_iter):
        if i<10:
            print(text)

    # # 8.5.2 Initializing the Model Parameters
    # npx.one_hot(np.array([0, 2]), len(vocab))
    #
    # X = np.arange(10).reshape((2, 5))
    # npx.one_hot(X.T, 28).shape
    #
    # # 8.5.3 RNN Model
    # num_hiddens = 512
    # device = d2l.try_gpu()
    # net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
    # state = net.begin_state(X.shape[0], device)
    # Y, new_state = net(X.as_in_context(device), state)
    # Y.shape, len(new_state), new_state[0].shape
    #
    # # 8.5.4 Prediction
    # predict_ch8('time traveller', 10, net, vocab, device)
    #
    # # 8.5.6 Trainging
    # num_epochs, lr = 500, 1
    # train_ch8(net, train_iter, vocab, lr, num_epochs, device)
    # plt.figure()
    # train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=True)
    pass


# 8.5.6 Trainging
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['trian'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, gluon.Block):
        net.initialize(ctx=device, force_reinit=True, init=init.Normal(0.01))
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0: animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or using random sampling
            state = net.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_context(device), y.as_in_context(device)
        with autograd.record():
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(net, 1)
        updater(batch_size=1)  # Since the `mean` function has been invoked
        metric.add(l * y.size, y.size)
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# 8.5.5 Gradient Clipping
def grad_clipping(net, theta):
    """Clip the gradient."""
    if isinstance(net, gluon.Block):
        params = [p.data() for p in net.collect_params().values()]
    else:
        params = net.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 8.5.4 Prediction
def predict_ch8(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: np.array([outputs[-1]], ctx=device).reshape((1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# 8.5.3 RNN Model
def init_rnn_state(batch_size, num_hiddens, device):
    return (np.zeros((batch_size, num_hiddens), ctx=device),)


def rnn(inputs, state, params):
    # Shape of `inputs`:(`num_steps`,`batch_size`,`vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`:(`batch_size`,`vocab_size`)
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)


class RNNModelScratch:
    """An RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)


# 8.5.2 Initializing the Model Parameters
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    # def normal(shape):
    #     return np.random.normal(scale=0.01, size=shape, ctx=device)

    normal = lambda shape: np.random.normal(scale=0.01, size=shape, ctx=device)
    zeros = lambda shape: np.zeros(shape, ctx=device)

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = zeros(num_hiddens)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = zeros(num_outputs)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
# 运行结束的提醒
beep_end()
show_figures()
