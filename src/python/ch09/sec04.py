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
@Desc       :   Sec 9.4 锚框（archor box）
@小结：
1.  以每个像素为中心，生成多个大小和宽高比不同的锚框
2.  交并比是两个边界框相交面积和相并面向之比
3.  在训练集中，为每个锚框标注两类标签：一类是锚框所含目标的类别；一类是真实边界框相对锚框的偏移量
4.  预测时，可以使用非极大值抑制来移除相似的预测边界框，从而令结果简洁
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np

from mxnet import autograd, gluon, init, nd, contrib, image
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures

np.set_printoptions(2)


# ----------------------------------------------------------------------
def main():
    # 目标检测算法会在输入图像中采样大量的区域，
    # 然后判断这些区域中是否包含感兴趣的目标，
    # 并且调整区域边缘从而更加准确地预测目标的真实边界框（ground-truth bounding box）
    # 不同的模型使用的区域采样方法可能不同，其中的一种方法：以每个像素为中心生成多个大小和宽高比不同的边界框，这些边界框称为锚框
    # 9.4.1 生成多个锚框
    # 假设输入图像高为 $h$，宽为 $w$，原图大小比 $s\in(0,1]$，生成框高宽比为 $r>0$
    # 得锚框的 $宽=ws\sqrt{r}$ 和 $高=hs/\sqrt{r}$
    # 如果设定好一组原图大小比 $s_1,s_2,\dot,s_n$ 和 一组生成框宽高比 $r_1,r_2,\dot,r_m$，那么会得到 $nm$ 个锚框，导致计算复杂度过高
    # 因此，只对包含 $s_1$ 和 $r_1$ 的组合感兴趣，即得到 $n+m-1$ 个锚框
    d2l.set_figsize()
    img = image.imread("../img/catdog.jpg").asnumpy()
    # d2l.show_images(img)

    h, w = img.shape[0:2]
    print("h=", h, "w=", w)
    X = nd.random.uniform(shape=(1, 3, h, w))
    Y = contrib.ndarray.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print("Y.shape(批量大小，锚框个数=wh(n+m-1)=561*728*(3+3-1)，4）=", Y.shape)

    # 将锚框集合转换成以每个像素为中心的锚框集合，n+m-1=3+3-1=5，每个锚框有4个元素描述坐标
    boxes = Y.reshape((h, w, 5, 4))
    print("锚框左上角 (x,y) 和 右下角 (x,y)=", boxes[250, 250, 0, :])
    print("锚框实际坐标值=", boxes[250, 250, 0, 0] * w, boxes[250, 250, 0, 1] * h, boxes[250, 250, 0, 2] * w, boxes[250, 250, 0, 3] * h)

    def show_bboxes(axes, bboxes, labels=None, colors=None):
        def _make_list(obj, default_values=None):
            if obj is None:
                obj = default_values
            elif not isinstance(obj, (list, tuple)):
                obj = [obj]
                pass
            return obj

        labels = _make_list(labels)
        colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]
            rect = d2l.bbox_to_rect(bbox.asnumpy(), color)
            axes.add_patch(rect)
            if labels and len(labels) > i:
                text_color = 'k' if color == 'w' else 'w'
                axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))
                pass
            pass
        pass

    d2l.set_figsize()
    bbox_scale = nd.array((w, h, w, h))
    # fig = d2l.plt.imshow(img)
    # show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale, ['s=0.75,r=1', 's=0.5,r=1', 's=0.25,r=1', 's=0.75,r=2', 's=0.75,r=0.5'])

    # 9.4.2 交并比
    # 使用 Jaccard 系数可以衡量两个集合的相似度，即衡量锚框和真实边界框之间的相似度
    # Jaccard 系数 $J(A,B)=\frac{|A\cap B|}{A\cup B|}$，即两个边界框相交面积与相并面积之比，也称为交并比（intersection over union, IoU）

    # 9.4.3 标注训练集的锚框
    # 在训练集中，
    # 每个锚框视为一个训练样本，
    # 每个训练样本标注两个标签：1. 锚框所含目标的类别；2. 真实边界模型相对锚框的领衔量

    ground_truth = nd.array([
        [0, 0.1, 0.08, 0.52, 0.92],
        [1, 0.55, 0.2, 0.9, 0.88]
    ])
    anchors = nd.array([
        [0, 0.1, 0.2, 0.3],
        [0.15, 0.2, 0.4, 0.4],
        [0.63, 0.05, 0.88, 0.98],
        [0.66, 0.45, 0.8, 0.8],
        [0.57, 0.3, 0.92, 0.9]
    ])
    # fig = d2l.plt.imshow(img)
    # show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    # show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
    labels = contrib.nd.MultiBoxTarget(anchors.expand_dims(axis=0), ground_truth.expand_dims(axis=0), nd.zeros((1, 3, 5)))
    # 返回的结果有三项
    print("labels.shape=", len(labels))
    # 第一项是为每个锚框标注的 4 个偏移量，其中负类锚框（类别为0）的偏移量标注为0
    print("每个锚框标注的 4 个偏移量".center(50,'-'))
    print("labels[0]=", labels[0])
    # 第二项是掩码变量，形状为（批量大小，锚框个数*4），用于过滤不关心的锚框的偏移量，按元素乘法，使之输出为 0
    print("掩码变量，形状为（批量大小，锚框个数*4）".center(50,'-'))
    print("labels[1]=", labels[1])
    # 第三项表示锚框标的类别（0表示背景，其它的按顺序：1表示狗，2表示猫）
    print("锚框标的类别（0表示背景，其它的按顺序：1表示狗，2表示猫）".center(50,'-'))
    print("labels[2]=", labels[2])
    # 第 0 个锚框虽然交并比最大的真实边界框的类别是狗，但是交并比小于阈值（默认为0.5），因此类别标注为背景
    # 第 4 个锚框虽然交并比最大的真实边界框的类别是猫，但是交并比小于阈值，因此类别标注为背景

    # 9.4.4 输出预测边界框
    # 使用 非极大值抑制（non-maximum suppression, NMS）移除相似的预测边界框，避免因为锚框数量较多时，同一个目标上可能会输出许多相似的预测边界框
    anchors = nd.array([
        [0.1, 0.08, 0.52, 0.92],
        [0.08, 0.2, 0.56, 0.95],
        [0.15, 0.3, 0.62, 0.91],
        [0.55, 0.2, 0.9, 0.88]
    ])
    offset_preds = nd.array([0] * anchors.size)
    cls_probs = nd.array([
        [0] * 4,  # 背景的预测概率
        [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
        [0.1, 0.2, 0.3, 0.9]  # 猫的预测概率
    ])
    # fig = d2l.plt.imshow(img)
    # show_bboxes(fig.axes, anchors * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

    output = contrib.nd.MultiBoxDetection(cls_probs.expand_dims(axis=0), offset_preds.expand_dims(axis=0), anchors.expand_dims(axis=0), nms_threshold=0.5)
    # 第一个元素是索引从0开始计数的预测类别（0为狗，1为猫），-1表示背景或在极大值抑制中被移除
    # 第二个元素是预测边界框的围住度
    # 剩余的四个元素表示预测边界框的坐标：左上角 (x,y) 和 右下角 (x,y)
    print("output=", output)

    fig = d2l.plt.imshow(img)
    for i in output[0].asnumpy():
        if i[0] == -1:
            continue
            pass
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, [nd.array(i[2:]) * bbox_scale], label)
        pass
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
