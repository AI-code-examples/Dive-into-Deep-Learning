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
@Desc       :   Sec 9.5 多尺度目标检测
@小结：
1.  可以在多个尺度下生成不同数量和不同大小的锚框，从而在多个惊讶下检测不同大小的目标
2.  特征图的形状能够确定任一图像上均匀采样的锚框中心
3.  使用输入图像在某个感受野区域内的信息来预测输入图像上与该区域相近的锚框的类别和偏移量
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd, contrib, image
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    img = data.get_image("data/img/catdog.jpg")
    h, w = img.shape[0:2]
    print(f"h={h},\tw={w}")
    d2l.set_figsize()

    def display_anchors(fmap_w, fmap_h, s):
        fmap = nd.zeros((1, 10, fmap_w, fmap_h))
        anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
        bbox_scale = nd.array((w, h, w, h))
        d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes, anchors[0] * bbox_scale)
        pass

    display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
    # display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
    # display_anchors(fmap_w=4, fmap_h=4, s=[0.15])

    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
