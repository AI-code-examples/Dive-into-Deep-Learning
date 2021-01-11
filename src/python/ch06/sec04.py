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
@Desc       :   Sec 6.4 循环神经网络的 Python 实现
@小结：
1.  可以用基于字符级循环神经网络的语言模型来生成文本序列
2.  当训练循环神经网络时，为了应对梯度爆炸，可以裁剪梯度
3.  困惑度是对交叉熵损失函数做指数运算后得到的值
@练习：
ToDo：各种试验都尝试了，对试验结果没有太多理解。
"""
import d2lzh as d2l
import math
import mxnet as mx
import time
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

from data import load_data_jay_lyrics
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
    print(nd.one_hot(nd.array([0, 2]), vocab_size))
    X = nd.arange(10).reshape((2, 5))
    print("X.shape=", X.shape, X)
    inputs = to_onehot(X, vocab_size)
    print(len(inputs), inputs[0], inputs[0].shape)

    show_subtitle("6.4.2 初始化模型参数")
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    ctx = d2l.try_gpu()

    def get_params():
        def _one(shape):
            return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

        # 隐藏层的参数
        W_xh = _one((num_inputs, num_hiddens))
        W_hh = _one((num_hiddens, num_hiddens))
        b_h = nd.zeros(num_hiddens, ctx=ctx)
        # 输出层的参数
        W_hq = _one((num_hiddens, num_outputs))
        b_q = nd.zeros(num_outputs, ctx=ctx)
        # 附着梯度
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.attach_grad()
            pass
        return params

    show_subtitle("6.4.3 定义模型")
    state = init_rnn_state(X.shape[0], num_hiddens, ctx)
    inputs = to_onehot(X.as_in_context(ctx), vocab_size)
    params = get_params()
    outputs, state_new = rnn(inputs, state, params)
    print(len(outputs), outputs[0].shape, state_new[0].shape)
    show_subtitle("6.4.4 定义预测函数")
    print(predict_rnn("分开", 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 32, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ["分开", "不分开"]
    show_subtitle("随机采样训练模型，并且创作歌词")
    train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx, True,
                                num_epochs, num_steps, lr, clipping_theta, batch_size,
                                pred_period, pred_len, prefixes)
    show_subtitle("顺序采样训练模型，并且创作歌词")
    train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx, False,
                                num_epochs, num_steps, lr, clipping_theta, batch_size,
                                pred_period, pred_len, prefixes)
    pass


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx,
                          corpus_indices, idx_to_char, char_to_idx, is_random_iter,
                          num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
        pass
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:
                # 如果使用随机采样，需要在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                # ToDo：如果使用顺序采样，需要使用 detach() 函数从计算图中分离隐藏状态(Ref:6.3)
                for s in state:
                    s.detach()
                    pass
                pass
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # outputs 有 num_steps 个形状为 (batch_size,vocab_size) 的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # outputs 在连结后形状为 (num_steps*batch_size,vocab_size) 的矩阵
                outputs = nd.concat(*outputs, dim=0)
                # Y 的形状为（ batch_size,num_steps) 的矩阵
                # 转置后成为 (batch_size*num_steps) 的向量
                # Y 的形状与 outputs 的行向量可以对应
                y = Y.T.reshape((-1,))
                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs, y).mean()
                pass
            l.backward()
            grad_clipping(params, clipping_theta, ctx)
            # ToDo：误差已经取过均值，梯度不需要再做平均
            d2l.sgd(params, lr, 1)
            l_sum += l.asscalar() * y.size
            n += y.size
            pass
        if (epoch + 1) % pred_period == 0:
            print("epoch %d, perplexity %f, tie %.2f sec" % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))
    pass


def grad_clipping(params, theta, ctx):
    """
    6.4.5 裁剪梯度函数：$\min(\frac{\theta}{||g||},1)g
    避免梯度误差或者梯度爆炸，将所有模型参数梯度的元素拼接成一个向量 g，裁剪后梯度的 $L_2$ 范数小于 $\theta$
    """
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
        pass
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            pass
        pass
    pass


def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        # H = nd.relu(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
        pass
    return outputs, (H,)


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        (Y, state) = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
            pass
        pass
    return ''.join([idx_to_char[i] for i in output])


def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
