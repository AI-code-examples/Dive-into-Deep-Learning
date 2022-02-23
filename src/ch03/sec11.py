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
@Desc       :   Sec 3.11 基于「多项式拟合实验」了解模型选择、欠拟合与过拟合
@小结：
1.  训练误差与泛化误差没有强因果关系，训练误差的持续减少不会导致泛化误差的持续减少
2.  增加验证数据集来进行模型选择
3.  欠拟合模型无法得到较低的训练误差
4.  过拟合模型无法得到较低的泛化误差
5.  模型的复杂度与现实模型越接近越好
6.  训练样本的数量越多越好
"""
import d2lzh as d2l

from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_figures


# ----------------------------------------------------------------------
def main():
    """
    func: $y=1.2 x-3.4 x^2+5.6 x^3 +5 + /epsilon$
    """
    n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
    features = nd.random.normal(shape=(n_train + n_test, 1))
    poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
    labels = (true_w[0] * poly_features[:, 0] +
              true_w[1] * poly_features[:, 1] +
              true_w[2] * poly_features[:, 2] +
              true_b)
    labels += nd.random.normal(scale=0.1, shape=labels.shape)

    def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsie=(3.5, 2.5)):
        d2l.plt.figure()
        d2l.set_figsize(figsie)
        d2l.plt.xlabel(x_label)
        d2l.plt.ylabel(y_label)
        d2l.plt.semilogy(x_vals, y_vals)
        if x2_vals and y2_vals:
            d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
            d2l.plt.legend(legend)

    num_epochs, loss = 100, gloss.L2Loss()

    def fit_and_plot(train_features, test_features, train_labels, test_labels):
        net = nn.Sequential()
        net.add(nn.Dense(1))
        net.initialize()
        batch_size = min(10, train_labels.shape[0])
        train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
        train_ls, test_ls = [], []
        for _ in range(num_epochs):
            for X, y in train_iter:
                with autograd.record():
                    l = loss(net(X), y)
                l.backward()
                trainer.step(batch_size)
            train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
            test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
        print('final epoch: train loss', train_ls[-1], ';\t', 'test loss', test_ls[-1])
        semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
        print('weight:', net[0].weight.data().asnumpy(), '\n',
              'bias:', net[0].bias.data().asnumpy())
        pass

    # 3. 正常拟合
    fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])

    # 4. 模型过于简单（欠拟合）
    fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])

    # 5. 训练样本不足（过拟合）
    fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2], labels[n_train:])

    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
