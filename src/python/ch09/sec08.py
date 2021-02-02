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
@Desc       :   Sec 9.8 区域卷积神经网络（R-CNN, Regionss with CNN features）系列
@小结：
1.  R-CNN 对图像选取若干提议区域，然后用卷积神经网络对每个提议区域做前向计算抽取特征，再用这些特征预测提议区域的类别和特征框
2.  Fast R-CNN 对 R-CNN 的主要改进：只对整个图像做卷积神经网络的前向计算。引入了兴趣区域池化层，从而令兴趣区域能够抽取出形状相同的特征
3.  Faster R-CNN 对 Fast R-CNN 的主要改进：将选择性搜索替换成区域提议网络，从而减少提议区域的生成数量，并且保证目标检测的精度
4.  Mask R-CNN 在 Faster R-CNN 基础上引入一个全卷积网络，从而借助目标的像素级位置进一步提升目标检测的精度
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # 9.8.1 R-CNN 的主要构成
    # 1. 对输入图像使用选择性搜索来选取多个高质量的提议区域。这些提议区域是在多个尺度下选取的，并且具有不同的形状和大小。每个提议区域将被标注类别和真实的边界框。
    # 2. 选取一个预训练的卷积神经网络，并且将其在输出层之前截断。将每个提议区域变形为网络需要的输入尺寸，并且通过前向计算输出抽取的提议区域特征。
    # 3. 将每个提议区域的特征连同其标注的类别作为一个样本，训练多个支持向量机对目标分类。其中每个支持向量机用来判断样本是否属于某一个类别。
    # 4. 将每个提议区域的特征连同其标注的边界框作为一个样本，训练线性回归模型来预测真实的边界框
    # 注：效果好，速度慢。

    # 9.8.2 Fast R-CNN 的主要改进：只对整个图像做卷积神经网络的前向计算
    X = nd.arange(16).reshape((1, 1, 4, 4))
    rois = nd.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
    nd.ROIPooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)

    # 9.8.3 Faster R-CNN 的主要改进：将选择性搜索替换成区域提议网络，从而减少提议区域的生成数量，并且保证目标检测的精度。

    # 9.8.4 Mask R-CNN 的主要改进：将兴趣区域池化层替换为兴趣区域对齐层，即通过双线性插值来保留特征图上的空间信息，从而更加适用于像素级预测。
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
