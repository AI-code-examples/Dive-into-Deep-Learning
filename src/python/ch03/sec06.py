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
@Desc       :   Sec 3.6 Softmax 回归的基本实现
@小结：
1.  使用 Softmax 回归做多分类问题
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, nd
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_epochs, lr = 5, 0.1
    loss = cross_entropy
    trainer = None
    params = [W, b]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += nd.array(y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f" % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        pass


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)  # 对 exp 矩阵同行元素求和
    return X_exp / partition


def test_softmax():
    X = nd.random.normal(shape=(2, 5))
    X_prob = softmax(X)
    print(X_prob, X_prob.sum(axis=1))


def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()


def test_pick():
    y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    y = nd.array([0, 2], dtype='int32')
    print(nd.pick(y_hat, y))


def accuracy(y_hat, y):
    return nd.array(y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


# ----------------------------------------------------------------------


if __name__ == '__main__':
    num_inputs = 784
    num_outputs = 10
    W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
    b = nd.zeros(num_outputs)
    W.attach_grad()
    b.attach_grad()
    main()
    # test_softmax()
    # test_pick()
    # 运行结束的提醒
    beep_end()
