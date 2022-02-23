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
@Desc       :   Sec 9.6 目标检测数据集（PIKACHU）
@小结：
1.  合成的 PIKACHU 数据集可用于测试目标检测模型
2.  目标检测的数据读取与图像分类的类似。然而，在引入边界框后，标签的形状和图像增广（例如：随机剪裁）发生的变化
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
import os
from mxnet import autograd, gluon, init, nd, image
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
from tools import beep_end, show_subtitle, show_title, show_figures


def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/')
    dataset = {
        'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
        'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
        'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'
    }
    for k, v in dataset.items():
        gutils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)
        pass
    pass


def load_data_pikachu(batch_size, edge_size=256):
    """

    @param batch_size:
    @param edge_size: 输出图像的宽和高
    """
    data_dir = data.get_root_path() + "data/pikachu"
    _download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # 输出图像的形状
        shuffle=True,  # 以随机顺序读取数据集
        rand_crop=1,  # 随机裁剪的概率为1
        min_object_covered=0.95,
        max_attempts=200
    )
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),
        shuffle=False
    )
    return train_iter, val_iter


# ----------------------------------------------------------------------
def main():
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_pikachu(batch_size, edge_size)
    batch = train_iter.next()
    print("（批量大小，通道数，高，宽）=", batch.data[0].shape)
    print("（批量大小，m,5）=", batch.label[0].shape)

    imgs = (batch.data[0][0:10].transpose((0, 2, 3, 1))) / 255
    axes = d2l.show_images(imgs, 2, 5).flatten()
    for ax, label in zip(axes, batch.label[0][0:10]):
        d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
        pass
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
