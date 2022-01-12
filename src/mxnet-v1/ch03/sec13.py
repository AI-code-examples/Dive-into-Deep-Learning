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
@Desc       :   Sec 3.13 基于权重丢弃法（Dropout）控制过拟合
@小结：
1.  丢弃法只应该在训练模型时使用
"""
import d2lzh as d2l

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_figures


# ----------------------------------------------------------------------
def main():
    # dropout_model()
    gluon_dropout_model()
    pass


def gluon_dropout_model():
    # Sec 3.13.3 基于 Gluon 简洁实现
    # 增加隐藏单元后，会导致更多信息丢失，也更容易过拟合
    net = create_4layers_net()
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)


def create_3layers_net():
    net = nn.Sequential()
    net.add(nn.Dense(num_hidden1, activation='relu'),
            nn.Dropout(drop_prob1),
            nn.Dense(num_hidden2, activation='relu'),
            nn.Dropout(drop_prob2),
            nn.Dense(num_outputs))
    return net


def create_4layers_net():
    num_hidden3 = 256
    drop_prob2, drop_prob3 = 0.3, 0.5
    net = nn.Sequential()
    net.add(nn.Dense(num_hidden1, activation='relu'),
            nn.Dropout(drop_prob1),
            nn.Dense(num_hidden2, activation='relu'),
            nn.Dropout(drop_prob2),
            nn.Dense(num_hidden3, activation='relu'),
            # nn.Dropout(drop_prob3),
            nn.Dense(num_outputs))
    return net


def dropout_model():
    # 1. 定义模型参数
    W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hidden1))
    b1 = nd.zeros(num_hidden1)
    W2 = nd.random.normal(scale=0.01, shape=(num_hidden1, num_hidden2))
    b2 = nd.zeros(num_hidden2)
    W3 = nd.random.normal(scale=0.01, shape=(num_hidden2, num_outputs))
    b3 = nd.zeros(num_outputs)
    params = [W1, b1, W2, b2, W3, b3]
    for param in params:
        param.attach_grad()

    # 2. 定义模型
    def net(X):
        X = X.reshape((-1, num_inputs))

        H1 = (nd.dot(X, W1) + b1).relu()
        if autograd.is_training():
            # 只在模型训练期间使用丢弃法
            H1 = dropout(H1, drop_prob1)
            pass

        H2 = (nd.dot(H1, W2) + b2).relu()
        if autograd.is_training():
            H2 = dropout(H2, drop_prob2)
            pass

        return nd.dot(H2, W3) + b3

    # 3. 训练和测试模型
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        # 参数全部丢弃
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob  # 随机生成掩码矩阵
    X = mask * X  # 提取应该返回的参数
    X = X / keep_prob  # 按比例调整参数大小
    return X


# ----------------------------------------------------------------------


if __name__ == '__main__':
    num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256
    # drop_prob1>drop_prob2 时，收敛速度变慢，测试精度容易振荡，模型容易欠拟合，但是训练时间足够长时，都可以找到最优结果
    # 可能是前面丢弃概率过大容易导致丢失的数据过多，需要更多次迭代才能将信息补全
    # drop_prob1, drop_prob2 = 0.5, 0.2
    drop_prob1, drop_prob2 = 0.2, 0.5
    num_epochs, lr, batch_size = 25, 0.5, 256
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
