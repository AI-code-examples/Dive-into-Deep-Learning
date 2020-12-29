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
@Desc       :   Sec 3.7 Softmax 回归的简洁实现
@小结：
1.  Gluon 的函数具有更好的数值稳定性
2.  Gluon 提供更简洁的 Softmax 回归
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, nd, gluon,init
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    num_epochs = 5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
