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
@Desc       :   Sec 3.12 基于权重衰减（正则化）控制过拟合
@小结：
1.  基于正则化方法为模型损失函数添加惩罚项使得模型参数值更小，从而应对过拟合问题
2.  权重衰减等价于 L2 范数正则化，使得学到的权重参数更接近0
3.  权重误差可以通过 Gluon 的 wd 超参数来指定
4.  可以定义多个 Trainer 实例对不同的模型参数使用不同的迭代方法
ToDo：为什么要对不同的模型参数使用不同的迭代方法？
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_figures


# ----------------------------------------------------------------------
def main():
    # 准备高维线性回归数据集
    # $y=0.05+\sum_{i=1}^p 0.01 x_i + \epsilon$
    n_train, n_test, num_inputs = 20, 100, 200
    true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05
    features = nd.random.normal(shape=(n_train + n_test, num_inputs))
    labels = nd.dot(features, true_w) + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    train_features, test_features = features[:n_train, :], features[n_train:, :]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    # 1. 初始化模型参数
    def init_params():
        w = nd.random.normal(scale=1, shape=(num_inputs, 1))
        b = nd.zeros(shape=(1,))
        w.attach_grad()
        b.attach_grad()
        return [w, b]

    def l2_penalty(w):
        return (w ** 2).sum() / 2

    batch_size, num_epochs, lr = 1, 100, 0.003
    net, loss = d2l.linreg, d2l.squared_loss
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)

    def fit_and_plot(lambd):
        w, b = init_params()
        train_ls, test_ls = [], []
        for _ in range(num_epochs):
            for X, y in train_iter:
                with autograd.record():
                    l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
                l.backward()
                d2l.sgd([w, b], lr, batch_size)
            train_ls.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
            test_ls.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
        d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                     range(1, num_epochs + 1), test_ls, ['train', 'test'])
        print('L2 norm of w:', w.norm().asscalar())
        pass

    fit_and_plot(lambd=0)
    fit_and_plot(lambd=3)

    def fit_and_plot_gluon(wd):
        net = nn.Sequential()
        net.add(nn.Dense(1))
        net.initialize(init.Normal(sigma=1))
        # ToDo：为什么自动使用 L2 范数的衰减方式？不能使用其他正则化方法？
        # 对权重参数衰减，权重名称以 weight 结尾
        trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd', {'learning_rate': lr, 'wd': wd})
        # 不对偏差参数衰减，偏差名称以 bias 结尾
        trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {'learning_rate': lr})
        train_ls, test_ls = [], []
        for _ in range(num_epochs):
            for X, y in train_iter:
                with autograd.record():
                    l = loss(net(X), y)
                l.backward()
                trainer_w.step(batch_size)
                trainer_b.step(batch_size)
            train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
            test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
        d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                     range(1, num_epochs + 1), test_ls, ['train', 'test'])
        print("L2 norm of w:", net[0].weight.data().norm().asscalar())
        pass

    fit_and_plot_gluon(0)
    fit_and_plot_gluon(3)

    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
