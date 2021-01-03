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
@Desc       :   Sec 5.6 深度卷积神经网络（AlexNet）
@小结：
1.  AlexNet 比 LeNet 增加了卷积层和更大的参数空间来拟合大规模数据集
2.  AlexNet 是深度神经网络的里程碑
3.  AlexNet 带来了观念上的转变
"""
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def mnist_alexnet():
    batch_size = 128
    # 出现 「out of memory」的错误信息时，可以减小 batch_size 或者 resize 的值
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    lr, num_epochs, ctx = 0.01, 5, d2l.try_gpu()
    net = create_net()
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
    pass


def main():
    # toy_alexnet()
    # Nvidia RTX 2060 比 Intel i7-9700 快 50 倍
    # 与 sec05(LeNet)对比可知，参数数量越多的模型，CPU与GPU计算的差距越大
    mnist_alexnet()
    pass


def load_data_fashion_mnist(batch_size, resize=None, root=d2l.os.path.join('-', '.mxnet', 'datasets', 'fashion-mnist')):
    root = d2l.os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
        pass
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if d2l.sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers
    )
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers
    )
    return train_iter, test_iter


def toy_alexnet():
    ctx = d2l.try_gpu()
    X = nd.random.uniform(shape=(1, 1, 224, 224), ctx=ctx)
    print("X.shape:\t", X.shape)
    net = create_net()
    net.initialize(ctx=ctx)
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)


def create_net():
    net = nn.Sequential()
    # 使用 11x11 窗口来捕获物体，使用步幅 4 来缩小输出的高与宽
    # 增加通道数来停留图片中的不同信息
    net.add(nn.Conv2D(32, kernel_size=11, strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2))
    # 使用 5x5 窗口来捕获细节，使用填充 2 来保持输入与输出的高与宽不变
    # 增加通道数来停留图片中的不同信息
    net.add(nn.Conv2D(128, kernel_size=5, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2))
    # 使用 3x3 窗口来捕获纹理，使用填充 1 来保持输入与输出的高与宽不变
    # 增加通道数来停留图片中的不同信息
    # 连续三层总面积层保证纹理信息能够被提取出来
    net.add(nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2))
    # 两个连续的全连接层可以捕获全局的信息
    # 使用丢弃层来缓解过拟合
    net.add(nn.Dense(2048, activation='relu'), nn.Dropout(0.5),
            nn.Dense(2048, activation='relu'), nn.Dropout(0.5))
    net.add(nn.Dense(10))
    return net


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
