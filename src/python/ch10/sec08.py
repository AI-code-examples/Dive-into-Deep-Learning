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
@Desc       :   Sec 10.8 基于 CNN 进行文本情感分类
@小结：
1.  可以使用一维卷积来表征时序数据
2.  多输入通道的一维互相关运算可以看作单输入通道的二维互相关运算
3.  时序最大池化层的输入在各个通道上的时间步数可以不同
4.  textCNN 主要使用了一维卷积层和时序最大池化层
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    X, K = nd.array([0, 1, 2, 3, 4, 5, 6]), nd.array([1, 2])
    print(corr1d(X, K))

    X = nd.array([
        [0, 1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 8]
    ])
    K = nd.array([
        [1, 2],
        [3, 4],
        [-1, -3]
    ])
    print(corr1d_multi_in(X, K))

    batch_size = 64
    print("开始下载数据...", end='')
    # d2l.download_imdb()
    print("下载完成！")

    print("开始载入数据...", end='')
    train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
    vocab = d2l.get_vocab_imdb(train_data)
    print("载入完成！")

    print("准备数据迭代器...", end='')
    train_iter = gdata.DataLoader(gdata.ArrayDataset(*d2l.preprocess_imdb(train_data, vocab)), batch_size, shuffle=True)
    test_iter = gdata.DataLoader(gdata.ArrayDataset(*d2l.preprocess_imdb(test_data, vocab)), batch_size)
    print("准备完成！")

    print("模型初始化...", end='')
    # 过于短的 kernel 反而效果不好
    embed_size, kernel_sizes, nums_channels = 100, [3, 5, 7, 11], [100, 100, 100, 100]
    ctx = d2l.try_gpu()
    net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
    net.initialize(init.Xavier(), ctx=ctx)
    print("初始化完成！")

    # 1. 加载预训练的词向量
    glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.constant_embedding.collect_params().setattr('grad_req', 'null')

    # 2. 训练模型
    lr, num_epochs = 0.001, 5
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

    # 3. 数据预测
    # 这里的字符串长度没有指定序列长度的补齐，当 kernel > len(pos_str) 时，系统会报错
    pos_str = ['this', 'movie', 'is', 'so', 'great', '', '', '', '', '', '']
    print(d2l.predict_sentiment(net, vocab, pos_str))
    neg_str = ['this', 'movie', 'is', 'so', 'bad', '', '', '', '', '', '']
    print(d2l.predict_sentiment(net, vocab, neg_str))

    pass


class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        # 参与训练的嵌入层
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # 时序最大池化层没有权重，后面会共用实例
        self.pool = nn.GlobalMaxPool1D()
        #  创建多个一维卷积层
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # 将两个嵌入层的输出按词向量连结
        # 嵌入层的形状为（批量大小，单词序列的长度，词向量的维度）
        embeddings = nd.concat(self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # 根据 Conv1D要求的输入格式，将词向量维（即一维卷积层的通道维）变换到前一维
        embeddings = embeddings.transpose((0, 2, 1))
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为（批量大小，通道大小，1）的NDArray
        # 使用 flatten() 去掉最后一维，然后在通道维上连结
        encoding = nd.concat(*[nd.flatten(self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


def corr1d(X, K):
    w = K.shape[0]
    Y = nd.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i:i + w] * K).sum()
    return Y


def corr1d_multi_in(X, K):
    # 沿着 X 和 K 的通道维（第0维）遍历
    # 使用 * 将结果列表变成 add_n() 的位置参数(positional argument) 进行相加
    multi_value = [corr1d(x, k) for x, k in zip(X, K)]
    print("多输入通道的多维输出".center(30, '-'))
    print(multi_value)
    return nd.add_n(*multi_value)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
