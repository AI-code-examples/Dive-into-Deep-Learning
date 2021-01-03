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
@Desc       :   Sec 5.10 批量归一化
@小结：
1.  数据标准化处理是：处理后的任一特征在数据集中所有样本上的均值为0、标准差为1。
2.  标准化处理输入数据使得各个特征的分布相近，从而更加容易训练出有效的模型。
3.  批量归一化：在模型训练时，利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使得整个神经网络在各个层的中间输出的数值更加稳定。
4.  对全连接层和卷积层进行批量归一化的方法稍有不同
5.  批量归一化层 和 丢弃层 一样，在训练模式下与预测模式下的计算方式是不同的
6.  使用 Gluon 提供的 BatchNorm 类更加简洁和方便
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    show_title("5.10.2 Python 代码实现")
    net = my_bn_lenet()
    train_net(net)

    show_title("5.10.3 Gluon 模块实现")
    net = gluon_bn_lenet()
    train_net(net)
    pass


def train_net(net):
    lr, num_epochs, batch_size, ctx = 1.0, 5, 256, d2l.try_gpu()
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
    print("net[1].gamma=", net[1].gamma.data().reshape((-1,)))
    print("net[1].beta=", net[1].beta.data().reshape((-1,)))


def gluon_bn_lenet():
    net = nn.Sequential()
    net.add(nn.Conv2D(6, kernel_size=5),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Conv2D(16, kernel_size=5),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Dense(120),
            nn.BatchNorm(),
            nn.Activation('sigmoid'))
    net.add(nn.Dense(84),
            nn.BatchNorm(),
            nn.Activation('sigmoid'))
    net.add(nn.Dense(10))
    return net


def my_bn_lenet():
    net = nn.Sequential()
    net.add(nn.Conv2D(6, kernel_size=5),
            BatchNorm(6, num_dims=4),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Conv2D(16, kernel_size=5),
            BatchNorm(16, num_dims=4),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Dense(120),
            BatchNorm(120, num_dims=2),
            nn.Activation('sigmoid'))
    net.add(nn.Dense(84),
            BatchNorm(84, num_dims=2),
            nn.Activation('sigmoid'))
    net.add(nn.Dense(10))
    return net


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过 autograd 来判断当前模式是训练模式还是预测模式
    if autograd.is_training():
        # 训练模式：则对输入数据进行归一化
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层时
            # 计算特征维上的均值和方差
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # 使用二维卷积层时
            # 计算通道维上（axis=1）的均值和方差
            # 需要保持 X 的形状方便后面进行广播运算
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            pass
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    else:
        # 预测模式：直接使用传入的移动平均所得的均值和方差
        mean = moving_mean
        var = moving_var
        pass
    # 对数据进行标准化
    # 训练模式：使用的是当前的均值和方差
    # 预测模式：使用的是传入的训练期间得到的移动平均的均值和方差
    X_hat = (X - mean) / nd.sqrt(var + eps)
    # 拉伸 与 平移
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var


class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims == 2:
            # 全连接层
            shape = (1, num_features)
        else:
            # 卷积层
            shape = (1, num_features, 1, 1)
            pass
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)
        pass

    def forward(self, X):
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
            pass
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma.data(), self.beta.data(), self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
