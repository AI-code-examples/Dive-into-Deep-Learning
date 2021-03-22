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
from mxnet import nd, np, npx, autograd, init, gluon
from mxnet.gluon import nn
from tools import beep_end, show_subtitle, show_title, show_figures
import matplotlib.pyplot as plt

npx.set_np()


# ----------------------------------------------------------------------
def main():
    T = 1000  # Generate a total of 1000 points
    time = np.arange(1, T + 1, dtype=np.float32)
    x = np.sin(0.01 * time) + np.random.normal(0, 0.2, (T,))
    plt.figure()
    d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

    tau = 4
    features = np.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i:T - tau + i]
    labels = d2l.reshape(x[tau:], (-1, 1))

    n_train = 600
    batch_size, num_epochs, lr = 16, 5, 0.1
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
    loss = gluon.loss.L2Loss()
    net = get_net()
    train(net, train_iter, loss, num_epochs, batch_size, lr)

    # 8.1.3 Prediction
    onestep_preds = net(features)
    plt.figure()
    d2l.plot([time, time[tau:]],
             [x.asnumpy(), onestep_preds.asnumpy()],
             'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))

    # 每次显示的结果不同
    multistep_preds = np.zeros(T)
    multistep_preds[:n_train + tau] = x[:n_train + tau]
    for i in range(n_train + tau, T):
        multistep_preds[i] = d2l.reshape(net(multistep_preds[i - tau:i].reshape(1, -1)), 1)
    plt.figure()
    x_steps = [time, time[tau:], time[n_train + tau:]]
    y_steps = [x.asnumpy(), onestep_preds.asnumpy(), multistep_preds[n_train + tau:].asnumpy()]
    d2l.plot(x_steps, y_steps, 'time', 'x', legend=['data', '1-step preds', 'multistep preds'], xlim=[1, 1000], figsize=(6, 3))

    max_steps = 64
    features = np.zeros((T - tau - max_steps + 1, tau + max_steps))
    for i in range(tau):
        features[:, i] = x[i:i + T - tau - max_steps + 1].T
    for i in range(tau, tau + max_steps):
        features[:, i] = d2l.reshape(net(features[:, i - tau:i]), -1)
    steps = (1, 4, 16, 64)
    plt.figure()
    x_steps = [time[tau + i - 1:T - max_steps + i] for i in steps]
    y_steps = [features[:, tau + i - 1].asnumpy() for i in steps]
    d2l.plot(x_steps, y_steps, 'time', 'x', legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000], figsize=(6, 3))
    pass


def get_net():
    # A simple MLP
    net = nn.Sequential()
    net.add(nn.Dense(10, activation='relu'),
            nn.Dense(1))
    net.initialize(init.Xavier())
    return net


def train(net, train_iter, loss, epochs, batch_size, lr):
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    for epoch in range(epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        print(f'epoch {epoch + 1}, loss:{d2l.evaluate_loss(net, train_iter, loss)}')


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
