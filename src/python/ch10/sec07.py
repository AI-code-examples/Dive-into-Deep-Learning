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
@Desc       :   Sec 10.7 基于 RNN 进行文本情感分类
@小结：
1.  文本分类把一段不定长的文本序列变换为文本的类别。
2.  可以应用预训练的词向量和循环神经网络对文本的情感进行分类
"""
import collections
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
import os
import random
import tarfile
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # 1. 读取数据集
    print("开始下载数据...", end='')
    # download_imdb()
    print("下载完成！")

    print("开始载入数据...", end='')
    train_data, test_data = read_imdb('train'), read_imdb('test')
    print("载入完成！")

    # 2. 预处理数据集
    vocab = get_vocab_imdb(train_data)
    print("# words in vocab:", len(vocab))

    # 3. 创建数据迭代器
    batch_size = 256    # 增加批处理的大小会提高精确度
    train_set = gdata.ArrayDataset(*preproccess_imdb(train_data, vocab))
    test_set = gdata.ArrayDataset(*preproccess_imdb(test_data, vocab))
    train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = gdata.DataLoader(test_set, batch_size)

    # 打印第一个批量数据的形状以及训练集中小批量的个数
    for X, y in train_iter:
        print('X.shape=', X.shape, '\t', 'y.shape=', y.shape)
        break
    print("#batches:", len(train_iter))

    embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_gpu()
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    net.initialize(init.Xavier(), ctx=ctx)

    # 1. 加载预训练的词向量
    glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.embedding.collect_params().setattr('grad_req', 'null')

    # 2. 训练模型
    # ToDo：有的时候训练会报错误：CUDNN_STATUS_INTERNAL_ERROR，也不知道什么原因就正确完成了。
    # 训练使用的时间参考似乎不正确
    lr, num_epochs = 0.01, 5
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

    # 3. 数据预测
    print(predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))
    print(predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad']))

    pass


def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=d2l.try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'


class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional 设为 True 得到双向循环神经网络
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers, bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs 的形状为（批量大小，每个输入序列的单词数目），
        # LSTM 需要将序列作为第一维，所以先将输入转置，再提取词特征
        # 输出形状为（每个输入序列的单词数目，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        # rnn.LSTM 只传入输入 embeddings，因此只返回最后一层的隐藏层在各个时间步的隐藏状态
        # outputs 的输出形状是（单词数目，批量大小，2*隐藏单元个数）
        outputs = self.encoder(embeddings)
        # 连续初始时间步和最终时间步的隐藏状态作为全连接层的输入
        # 输出形状为（批量大小，4*隐藏单元个数）
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs


def download_imdb(data_dir='../data'):
    url = ('http://ai.standord.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)


def read_imdb(folder='train'):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('../data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


def get_tokenized_imdb(data):
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)


def preproccess_imdb(data, vocab):
    def pad(x):
        max_l = 500  # 将每条评论通过截断或者补0，使得长度变成 500
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
