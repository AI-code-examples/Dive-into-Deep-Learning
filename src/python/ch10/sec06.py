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
@Desc       :   Sec 10.6 求近义词和类比词
@小结：
1.  在大规模语料上预训练的词向量常常可以应用于下游自然语言处理任务中
2.  可以应用预训练的词向量求近义词和类比词
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
    glove_6b50d = text.embedding.create('glove', pretrained_file_name='glove.6B.50d.txt')

    # how_to_use_pretrained_model(glove_6b50d)

    # 10.6.2 应用预训练的词向量
    # 1. 求近义词
    def knn(W, x, k):
        # knn(k-nearest neighbor) K-近邻算法求近义词
        epsilon = 1e-9  # 用于保证数值的稳定性
        cos = nd.dot(W, x.reshape((-1,))) / ((nd.sum(W * W, axis=1) + epsilon).sqrt() * nd.sum(x * x).sqrt())
        topk = nd.topk(cos, k=k, ret_typ='indices').asnumpy().astype('int32')
        return topk, [cos[i].asscalar() for i in topk]

    def get_similar_tokens(query_token, k, embed):
        topk, cos = knn(embed.idx_to_vec, embed.get_vecs_by_tokens([query_token]), k + 1)
        for i, c in zip(topk[1:], cos[1:]):
            # 除去输入词 topk[0] 与 cos[0]
            print("cosine sim=%.3f: %s" % (c, (embed.idx_to_token[i])))

    get_similar_tokens('chip', 3, glove_6b50d)
    get_similar_tokens('baby', 3, glove_6b50d)
    get_similar_tokens('beautiful', 3, glove_6b50d)

    def get_analogy(token_a, token_b, token_c, embed):
        vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
        x = vecs[1] - vecs[0] + vecs[2]
        topk, cos = knn(embed.idx_to_vec, x, 1)
        return embed.idx_to_token[topk[0]]

    print(get_analogy('man', 'woman', 'son', glove_6b50d))
    print(get_analogy('beijing', 'china', 'tokyo', glove_6b50d))
    print(get_analogy('bad', 'worst', 'big', glove_6b50d))
    print(get_analogy('do', 'did', 'go', glove_6b50d))

    pass


def how_to_use_pretrained_model(glove_6b50d):
    # 10.6.1 使用预训练的词向量
    print(text.embedding.get_pretrained_file_names().keys())
    print(text.embedding.get_pretrained_file_names('glove'))
    print(len(glove_6b50d))
    print(glove_6b50d.token_to_idx['beautiful'])
    print(glove_6b50d.idx_to_token[3367])


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
