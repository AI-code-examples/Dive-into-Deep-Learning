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
@Desc       :   Sec 9.1 图像增广
@小结：
1.  图像增广基于现有训练数据生成随机图像从而应对过拟合
2.  为了在预测时得到确定的结果，通常只将图像增广应用在训练样本上，而不在预测时使用含随机操作的图像增广
3.  可以从 Gluon 的 transforms 模块中获取有关图片增广的类
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
import time
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # 9.1.1 常用的图像增广方法
    d2l.set_figsize()
    img = data.get_image("data/img/cat1.jpg")
    d2l.plt.imshow(img.asnumpy())

    # 1. 翻转和裁剪
    # flipping_and_cropping(img)

    # 2. 变化颜色
    # change_the_color(img)

    # 3. 叠加多个图像增广方法
    # multiple_image_augmetation(img)

    # 9.1.2 使用图像增广训练模型
    # gdata.vision.CIFAR10() 使用的路径是 C:\Users\ygpfr\AppData\Roaming\mxnet\datasets\cifar10\
    # 显示 CIFAR-10 数据集中前 32 张训练图像
    show_images(gdata.vision.CIFAR10(train=True)[0:32][0], 4, 8, scale=0.8)

    import sys
    num_workers = 0 if sys.platform.startswith('win32') else 4

    def load_cifar10(is_train, augs, batch_size):
        return gdata.DataLoader(
            gdata.vision.CIFAR10(train=is_train).transform_first(augs),
            batch_size=batch_size, shuffle=is_train, num_workers=num_workers
        )

    def train_with_data_aug(train_augs, test_augs, lr=0.001):
        batch_size, ctx, net = 256, try_all_gpus(), d2l.resnet18(10)
        net.initialize(ctx=ctx, init=init.Xavier())
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
        loss = gloss.SoftmaxCrossEntropyLoss()
        train_iter = load_cifar10(True, train_augs, batch_size)
        test_iter = load_cifar10(True, test_augs, batch_size)
        train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=30)
        pass

    tensor_aug = gdata.vision.transforms.ToTensor()
    no_aug = gdata.vision.transforms.Compose([tensor_aug])
    # ToDo: 有增广训练的效果 不如 无增广训练的效果？
    print("无增广的训练结果".center(50, '*'))
    train_with_data_aug(no_aug, no_aug)

    flip_aug = gdata.vision.transforms.RandomFlipLeftRight()
    flip_augs = gdata.vision.transforms.Compose([flip_aug, tensor_aug])

    print("有增广的训练结果".center(50, '*'))
    train_with_data_aug(flip_augs, no_aug)

    color_aug = gdata.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    shape_aug = gdata.vision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    # 四个增广方法生成的数据对于 RTX 2060 显存不够
    # augs = gdata.vision.transforms.Compose([flip_aug, color_aug, shape_aug, tensor_aug])
    augs = gdata.vision.transforms.Compose([flip_aug, color_aug, shape_aug, tensor_aug])
    # ToDo: 其他增广方法模型无法正常训练？
    # print("增广集的训练结果".center(50, '*'))
    # train_with_data_aug(shape_aug, no_aug)
    pass


def try_all_gpus():
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
        pass
    return ctxes


def evaluate_accuracy(data_iter, net, ctx=None):
    # 与 3.6节 和 5.5节 中定义的函数不同，这个定义更加通用
    # 通过辅助函数 _get_batch() 使用 ctx 变量所包含的所有 GPU 来评价模型
    if ctx is None:
        ctx = [mx.cpu()]
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
        pass
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
            pass
        acc_sum.wait_to_read()
        pass
    return acc_sum.asscalar() / n


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print("training on", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
        pass
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
                pass
            for l in ls:
                l.backward()
                pass
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
            pass
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec" %
              (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start))
        pass
    pass


def _get_batch(batch, ctx):
    # 辅助函数，将小批量数据样本 batch 划分并且复制到 ctx 变量指定的各个显存上
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
        pass
    return (
        gutils.split_and_load(features, ctx),
        gutils.split_and_load(labels, ctx),
        features.shape[0]
    )


def multiple_image_augmetation(img):
    flip_aug = gdata.vision.transforms.RandomFlipLeftRight()
    color_aug = gdata.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    shape_aug = gdata.vision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    augs = gdata.vision.transforms.Compose([flip_aug, color_aug, shape_aug])
    apply(img, augs)


def change_the_color(img):
    # 将图像的亮度随机变化为原图亮度的 50%
    apply(img, gdata.vision.transforms.RandomBrightness(0.5))
    # 将图像的色调随机变化为原图色调的 50%
    apply(img, gdata.vision.transforms.RandomHue(0.5))
    # 将图像的对比度随机变化为原图对比度的 50%
    apply(img, gdata.vision.transforms.RandomContrast(0.5))
    # 将图像的饱和度随机变化为原图饱和度的 50%
    apply(img, gdata.vision.transforms.RandomSaturation(0.5))
    # 同时随机调整图像颜色的各种属性
    color_aug = gdata.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    apply(img, color_aug)


def flipping_and_cropping(img):
    # 一半概率的图像左右翻转
    apply(img, gdata.vision.transforms.RandomFlipLeftRight())
    # 一半概率的图像上下翻转
    apply(img, gdata.vision.transforms.RandomFlipTopBottom())
    # 随机裁剪
    shape_aug = gdata.vision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    apply(img, shape_aug)


def show_images(imgs, num_rows, num_cols, scale=2.0):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
            pass
        pass
    return axes


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    # 对输入图像 img 多次运行图像增广方法 aug，并展示所有的结果
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
