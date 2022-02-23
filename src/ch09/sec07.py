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
@Desc       :   Sec 9.7 单发多框检测（SSD）
@小结：
1.  单发多框检测是一个多尺度的目标检测模型。这个模型基于基础网络块和各个多尺度特征块生成不同数量和不同大小的锚框，并且通过预测锚框的类别和偏移量检测不同大小的目标。
2.  单发多框检测在训练中根据类别和偏移量的预测和标注值计算损失函数。
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
import sec06
import time
from mxnet import autograd, gluon, init, nd, contrib, image
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


def cls_predictor(num_anchors, num_classes):
    # 1. 类别预测层
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_anchors):
    # 2. 边界框预测层
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    # 3. 连结多尺度的预测
    block.initialize()
    return block(x)


def flatten_pred(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()


def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(num_channels):
    # 4. 高和宽减半模块：使得输出特征图中每个单元的感受野变得更加广阔
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(
            nn.Conv2D(num_channels, kernel_size=3, padding=1),
            nn.BatchNorm(in_channels=num_channels),
            nn.Activation('relu')
        )
        pass
    blk.add(nn.MaxPool2D(2))
    return blk


def base_net():
    # 5. 基础网络块：用来从原始图像中抽取特征
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
        pass
    return blk


def get_blk(i):
    # 6. 完整的模型
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
        pass
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = contrib.nd.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


class TinySSD(nn.Block):
    def __init__(self, num_classes, num_anchors, sizes, ratios, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.sizes = sizes
        self.ratios = ratios
        for i in range(5):
            # 批量赋值语句：self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))
            pass
        pass

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), self.sizes[i], self.ratios[i], getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )
            pass
        return nd.concat(*anchors, dim=1), concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)), concat_preds(bbox_preds)


# ----------------------------------------------------------------------
def main():
    print("3. 连结多尺度的预测".center(50, '='))
    Y1 = forward(nd.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
    Y2 = forward(nd.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
    print(Y1.shape, Y2.shape, concat_preds([Y1, Y2]).shape)

    print("4. 高和宽的减半模型".center(50, '='))
    print(forward(nd.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape)

    print("5. 基础网络块".center(50, '='))
    print(forward(nd.zeros((2, 3, 256, 256)), base_net()).shape)

    print("6. 完整的模型".center(50, '='))
    sizes = [
        [0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]
    ]
    ratios = [[1, 2, 0.5]] * 5
    num_classes = 1
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    net = TinySSD(num_classes, num_anchors, sizes, ratios)
    net.initialize()
    X = nd.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)

    print("output anchors:", anchors.shape)
    print("output class preds:", cls_preds.shape)
    print("output bbox preds:", bbox_preds.shape)

    batch_size = 32
    train_iter, _ = d2l.load_data_pikachu(batch_size)
    ctx, net = d2l.try_gpu(), TinySSD(num_classes, num_anchors, sizes, ratios)
    net.initialize(init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2, 'wd': 5e-4})
    cls_loss = gloss.SoftmaxCrossEntropyLoss()
    bbox_loss = gloss.L1Loss()

    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        # 2. 定义损失函数
        # 目标检测有两个损失：
        #   -   有关锚框类别的损失，即交叉熵损失函数
        #   -   正类锚框偏移量的损失。
        #       -   预测偏移量是个回归问题，没有使用平方损失函数，而是 L_1 损失。
        #       -   掩码变量 bbox_masks 保证负类锚框和填充锚框不参与损失的计算。
        # 两个损失相加得到模型的最终损失函数
        cls = cls_loss(cls_preds, cls_labels)
        bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
        return cls + bbox

    def cls_eval(cls_preds, cls_labels):
        # 2. 定义评价函数
        return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()

    def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
        # 2. 定义评价函数
        return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()

    # 3. 训练模型
    for epoch in range(20):
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        train_iter.reset()
        start = time.time()
        for batch in train_iter:
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                anchors, cls_preds, bbox_preds = net(X)
                bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(anchors, Y, cls_preds.transpose((0, 2, 1)))
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
                pass
            l.backward()
            trainer.step(batch_size)
            acc_sum += cls_eval(cls_preds, cls_labels)
            n += cls_labels.size
            mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            m += bbox_labels.size
            pass
        if (epoch + 1) % 5 == 0:
            print("epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec" % (epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))
            pass
        pass

    # 9.7.3 预测目标
    img = image.imread('../img/pikachu.jpg')
    feature = image.imresize(img, 256, 256).astype('float32')
    X = feature.transpose((2, 0, 1)).expand_dims(axis=0)

    def predict(X):
        anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
        cls_probs = cls_preds.softmax().transpose((0, 2, 1))
        output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
        return output[0, idx]

    output = predict(X)
    d2l.set_figsize((5, 5))

    def display(img, output, threshold):
        fig = d2l.plt.imshow(img.asnumpy())
        for row in output:
            score = row[1].asscalar()
            if score < threshold:
                continue
                pass
            h, w = img.shape[0:2]
            bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
            d2l.show_bboxes(fig.axes, bbox, "%.2f" % score, 'w')
            pass
        pass

    display(img, output, threshold=0.1)

    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
