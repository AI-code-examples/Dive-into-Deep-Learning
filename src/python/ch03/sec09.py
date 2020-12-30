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
@Desc       :   Sec 3.9 多层感知机的代码实现
@小结：
"""
import d2lzh as d2l

from mxnet import autograd, nd
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
    b1 = nd.zeros(num_hiddens)
    W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
    b2 = nd.zeros(num_outputs)
    params = [W1, b1, W2, b2]
    for param in params:
        param.attach_grad()

    def relu(X):
        return nd.maximum(X, 0)

    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(nd.dot(X, W1) + b1)
        return nd.dot(H, W2) + b2

    from mxnet.gluon import loss as gloss
    loss = gloss.SoftmaxCrossEntropyLoss()
    num_epochs, lr = 15, 0.5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
