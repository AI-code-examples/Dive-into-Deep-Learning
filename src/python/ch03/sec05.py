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
@Desc       :   Sec 3.5 图像分类数据集（Fashion-MNIST)
@小结：
1.  Fashion-MNIST 是 10 分类的服饰数据集
2.  高和宽分别为 $h$ 和 $w$ 像素的图像的形状记为：$h\times w$ 或 $(h,w)$
"""
from mxnet import nd
from tools import beep_end, show_subtitle
from mxnet.gluon import data as gdata
import sys
import time
from matplotlib import pyplot as plt
import d2lzh as d2l


# ----------------------------------------------------------------------
def main():
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)
    # print_dataset(mnist_test, mnist_train)
    # show_images(mnist_train)
    read_dataset(mnist_train)
    pass


def read_dataset(mnist_train):
    batch_size = 256
    transformer = gdata.vision.transforms.ToTensor()
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=8)
    start = time.time()
    for X, y in train_iter:
        continue
    end = time.time()
    print("read train data consume: %.2f sec" % (end - start))


def show_images(mnist_train):
    X, y = mnist_train[0:9]
    show_fashion_mnist(X, get_fashion_mnist_labels(y))


def print_dataset(mnist_test, mnist_train):
    print(len(mnist_train))
    print(len(mnist_test))
    feature, label = mnist_train[0]
    print(feature.shape, feature.dtype)
    print(label, type(label), label.dtype)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    if len(plt.get_fignums()) > 0:
        plt.show()
