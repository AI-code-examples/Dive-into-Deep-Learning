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
@Desc       :   Sec 7.3 小批量随机梯度下降
@小结：
1.  小批量随机梯度每次随机均匀采样一个小批量的训练样本来计算梯度
2.  （小批量）随机梯度下降的学习率可以在迭代过程中自我衰减
3.  小批量随机梯度在每个迭代周期的耗时介于梯度下降和随机梯度下降的耗时之间
"""
import time

import d2lzh as d2l
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

from data import get_data_ch7
from tools import beep_end, show_figures


# ----------------------------------------------------------------------
def main():
    features, labels = get_data_ch7()
    print(features.shape)
    print(labels.shape)
    # train_sgd(sgd, None, 1, features, labels, 1500, 6)  # 梯度下降
    # train_sgd(sgd, None, 0.005, features, labels, 1)  # 随机梯度下降
    # train_sgd(sgd, None, 0.05, features, labels, 10)  # 小批量随机梯度下降
    train_gluon_ch7('sgd', {'learning_rate': 0.05}, features, labels, 10)
    pass


def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels, batch_size=10, num_epochs=2):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()

    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), trainer_name, trainer_hyperparams)
    start = time.time()
    for _ in range(num_epochs):
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
                pass
            l.backward()
            trainer.step(batch_size)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
                pass
            pass
        pass
    print("loss: %f, %f sec per epoch" % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    pass


def train_sgd(trainer_fn, states, lr, features, labels, batch_size, num_epochs=2):
    train_ch7(trainer_fn, states, {'lr': lr}, features, labels, batch_size, num_epochs)


def train_ch7(trainer_fn, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    net, loss = d2l.linreg, d2l.squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w, b), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    start = time.time()
    for _ in range(num_epochs):
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()  # 使用平均损失
                pass
            l.backward()
            trainer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每 100 个样本记录下当前训练误差
                pass
            pass
        pass
    print("loss: %f, %f sec per epoch" % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel("epoch")
    d2l.plt.ylabel("loss")
    pass


def sgd(params, states, hyperparams):
    """

    """
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
        pass
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
