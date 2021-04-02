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

from d2l import mxnet as d2l
from mxnet import nd, np, npx, autograd, gluon
from mxnet.gluon import nn
from tools import beep_end, show_subtitle, show_title, show_figures

npx.set_np()


# ----------------------------------------------------------------------
def main():
    n_train = 50  # No. of training examples
    x_train = np.sort(np.random.rand(n_train) * 5)  # Training inputs

    # 10.2.1. Generating the Dataset
    # 人工数据集：$y_i=2\sin(x_i)+x_i^{0.8}+\epsilon$
    f = lambda x: 2 * np.sin(x) + x ** 0.8
    y_train = f(x_train) + np.random.normal(0.0, 0.5, (n_train,))  # Training outputs
    x_test = np.arange(0, 5, 0.1)  # Testing examples
    y_truth = f(x_test)  # Ground-truth outputs for the testing examples
    n_test = len(x_test)

    def plot_kernel_reg(y_hat):
        d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
        d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

    # 10.2.2. Average Pooling
    y_hat = y_train.mean().repeat(n_test)
    plot_kernel_reg(y_hat)

    # 10.2.3. Nonparametric Attention Pooling
    # `X_repeat` shape: (`n_test`, `n_train`)
    # each row contains the same testing inputs(i.e., same queries)
    X_repeat = x_test.repeat(n_train).reshape((-1, n_train))
    # `x_train` contains the keys.
    # `attention_weights` shape: (`n_test`, `n_train`)
    # each row contains attention weights to be assigned among the values(`y_train`) given each query
    attention_weights = npx.softmax(-(X_repeat - x_train) ** 2 / 2)
    # `y_hat` is weighted average of values, where weights are attention weights
    y_hat = np.dot(attention_weights, y_truth)
    plt.figure()
    plot_kernel_reg(y_hat)
    d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0), xlabel='Sorted training inputs', ylabel='Sorted testing inputs')

    # 10.2.4. Parametric Attention Pooling
    X = np.ones((2, 1, 4))
    Y = np.ones((2, 4, 6))
    print(npx.batch_dot(X, Y).shape)  # Batch Matrix Multiplication

    weights = np.ones((2, 10)) * 0.1
    values = np.arange(20).reshape((2, 10))
    print(npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1)))

    X_tile = np.tile(x_train, (n_train, 1))
    Y_tile = np.tile(y_train, (n_train, 1))
    keys = X_tile[(1 - np.eye(n_train)).astype('bool')].reshape((n_train, -1))
    values = Y_tile[(1 - np.eye(n_train)).astype('bool')].reshape((n_train, -1))

    net = NWKernelRegression()
    net.initialize()
    loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
    for epoch in range(50):
        with autograd.record():
            l = loss(net(x_train, keys, values), y_train)
        l.backward()
        trainer.step(1)
        print(f'epoch {epoch + 1}, loss{float(l.sum()):.6f}')
        animator.add(epoch + 1, float(l.sum()))

    keys = np.tile(x_train, (n_test, 1))
    values = np.tile(y_train, (n_test, 1))
    y_hat = net(x_test, keys, values)
    plot_kernel_reg(y_hat)
    d2l.show_heatmaps(np.expand_dims(np.expand_dims(net.attention_weights, 0), 0), xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
    pass


class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super(NWKernelRegression, self).__init__(**kwargs)
        self.attention_weights = None
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # queries shape: (no. of queries, no. of key-value pairs)
        queries = queries.repeat(keys.shape[1]).reshape((-1, keys.shape[1]))
        # attention_weights shape: (no. of queries, no. of key-value pairs)
        self.attention_weights = npx.softmax(-((queries - keys) * self.w.data()) ** 2 / 2)
        # values shape: (no. of queries, no. of key-value pairs)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1), np.expand_dims(values, -1)).reshape(-1)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
