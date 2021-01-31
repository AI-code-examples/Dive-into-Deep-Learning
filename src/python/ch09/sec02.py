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
@Desc       :   Sec 9.2 迁移学习的常用技术——微调
@小结：
1.  迁移学习将从源数据集学到的知识迁移到目标数据集上。
    -   微调是迁移学习的一种常用技术
2.  目标模型复制了源模型上除了输出层外的所有模型设计及其参数，并且基于目标数据集微调这些参数，而目标模型的输出层需要从头训练
3.  微调参数会使用比较小的学习率，而从头训练的输出层使用比较大的学习率
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
import os
import zipfile
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, utils as gutils
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # 微调的四个步骤：
    # 1.    在源数据集上预训练一个神经网络模型，即源模型
    # 2.    创建一个新的神经网络模型，即目标模型（复制了源模型中除输出层外的所有模型设计及其参数）
    # 3.    为目标模型添加一个输出大小为目标数据集类别个数的输出层，并且随机初始化该层的模型参数
    # 4.    在目标数据集上训练目标模型（输出层的参数在训练过程中获得，其余层的参数都是基于源模型的参数微调得到）
    data_dir = data.get_root_path() + "data/"
    base_url = "https://apache-mxnet.s3-accelerate.amazonaws.com/"
    fname = gutils.download(base_url + "gluon/dataset/hotdog.zip", path=data_dir, sha1_hash="fba480ffa8aa7e0febbb511d181409f899b9baa5")
    with zipfile.ZipFile(fname, 'r') as z:
        z.extractall(data_dir)
        pass
    train_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, "hotdog/train"))
    test_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, "hotdog/test"))
    hotdogs = [train_imgs[i][0] for i in range(8)]
    not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

    # 训练时，先从图像中裁剪出随机大小和随机高宽比的一块随机区域，然后将该区域缩放为 224*224 作为输入
    # 测试时，先将图像缩放为 256*256，然后从中裁剪出 224*224 的中心区域作为输入
    # 指定 RGB 三个通道的均值和方差来将图像通道归一化
    normalize = gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.RandomResizedCrop(224),
        gdata.vision.transforms.RandomFlipLeftRight(),
        gdata.vision.transforms.ToTensor(),
        normalize
    ])
    test_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.Resize(256),
        gdata.vision.transforms.CenterCrop(224),
        gdata.vision.transforms.ToTensor(),
        normalize
    ])
    # 预训练的源模型实例含有两个成员变量：
    # -   features：包含模型除输出层以外的所有层
    # -   output：包含模型的输出层
    # 下载文件： https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/resnet18_v2-a81db45f.zip
    # 解压到目录：C:\Users\ygpfr\AppData\Roaming\mxnet\models
    pretained_net = model_zoo.vision.resnet18_v2(pretrained=True)
    print("ResNet 的输出层".center(50, '*'))
    print(pretained_net.output)

    # 创建目标模型 finetune_net，定义与预训练模型一样，最后的输出个数等于目标数据集的类别数
    finetune_net = model_zoo.vision.resnet18_v2(classes=2)
    finetune_net.features = pretained_net.features
    finetune_net.output.initialize(init.Xavier())
    # 由于 features 中的模型参数是在 ImageNet 数据集上预训练得到的，已经足够好了，因此只需要使用较小的学习率来微调这些参数
    # 然后 output 中的模型参数是随机初始化的，因此需要更大的学习率从头开始
    # output 中的模型参数将在迭代中使用相对于Trainer实例中的学习率的 10 倍大的学习率
    finetune_net.output.collect_params().setattr('lr_mult', 10)

    # 3. 微调模型
    def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=15):
        train_iter = gdata.DataLoader(train_imgs.transform_first(train_augs), batch_size, shuffle=True)
        test_iter = gdata.DataLoader(test_imgs.transform_first(test_augs), batch_size)
        ctx = d2l.try_gpu()
        net.collect_params().reset_ctx(ctx)
        net.hybridize()
        loss = gloss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': 0.001})
        d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
        pass

    train_fine_tuning(finetune_net, 0.01)

    scratch_net = model_zoo.vision.resnet18_v2(classes=2)
    scratch_net.initialize(init.Xavier())
    train_fine_tuning(scratch_net, 0.1)

    # 微调模型因为参数初始值更好，往往在相同的迭代周期下取得更高的精度
    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
