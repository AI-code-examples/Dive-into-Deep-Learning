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
@Desc       :   Sec 10.3 Word2Vec 的实现
@小结：
1.  可以使用 Gluon 通过负采样训练跳字模型
2.  二次采样试图尽可能减轻高频词对训练词嵌入模型的影响
3.  可以将长度不同的样本填充至长度相同的小批量，并且通过掩码变量区分填充与非填充，然后只令非填充参与损失函数的计算
"""
import collections
import d2lzh as d2l
import data
import math
import mxnet as mx
import numpy as np
import random
import sys
import time
import zipfile
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # 10.3.1 提取预处理数据集
    # PTB(Penn Tree Bank) 常用的小型语料库。采样自《华尔街日报》的文章，包括：训练集、验证集和测试集。
    # with zipfile.ZipFile("../data/ptb.zip", 'r') as zin:
    #     zin.extractall("../data/")
    #     pass
    with open("../data/ptb/ptb.train.txt", 'r') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]
        pass
    print("# 原始的训练数据集中共有句子: %d" % len(raw_dataset))
    # 输出数据集的前 3 个句子
    # 数据集中句尾符为「<cos>」（没看到？），生僻词为「<unk>」，数字为「N」
    for st in raw_dataset[:3]:
        print('# 句子：', st)
        print("# tokens:", len(st), st[:5])
        pass

    # 1 建立词语索引
    # 为了简化计算，这里只保留了数据集中至少出现 5 次的词
    # 具体在应用中保留的词量根据实际情况调整
    counter = collections.Counter([tk for st in raw_dataset for tk in st])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

    # 将词映射到整数索引
    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in raw_dataset]
    num_tokens = sum([len(st) for st in dataset])
    print("# 词典中单词的个数: %d" % num_tokens)

    # 2. 二次采样
    # 在一个背景窗口中，某个词（例如：chip）和另一个低频词（例如：microprocessor）同时出现的意义比和另一个高频词（例如：the）同时出现的意义更大。
    # 因此，训练词嵌入模型前可以对词进行二次采样，即对数据集中的被索引词按照一定的概率丢弃，频率越高的词丢弃的概率越大
    # 丢弃概率的公式：$P(w_i)=\max(1-\sqrt{\frac{t}{f(w_i)}},0)$，$f(w_i)$ 为数据集中单词 $w_i$ 的个数与数据集中单词个数之比，常数 $t$ 是一个超参数（常设为 1e-4）
    def discard(idx):
        return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * num_tokens)

    # 二次采样后的数据集
    subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
    print("# 二次采样后词典中单词的个数: %d" % sum([len(st) for st in subsampled_dataset]))

    def compare_counts(token):
        # 二次采样前单词出现的频率 与 二次采样后单词出现的频率
        return "# %s: before=%d, after=%d" % (
            token,
            sum([st.count(token_to_idx[token]) for st in dataset]),
            sum([st.count(token_to_idx[token]) for st in subsampled_dataset])
        )

    print(compare_counts('the'))
    print(compare_counts('join'))

    # 3. 提取中心词和背景词
    def get_centers_and_contexts(dataset, max_window_size):
        centers, contexts = [], []
        for st in dataset:
            # 每个句子至少要有 2 个词才可能组成一对「中心词-背景词」
            if len(st) < 2:
                continue
                pass
            centers += st
            for center_i in range(len(st)):
                window_size = random.randint(1, max_window_size)
                indices = list(range(max(0, center_i - window_size), min(len(st), center_i + 1 + window_size)))
                # 将中心词排除在背景词之外
                indices.remove(center_i)
                contexts.append([st[idx] for idx in indices])
                pass
            pass
        return centers, contexts

    # 模拟数据集：两个句子，分别含有词数 7 和 3，设最大背景窗口为 2
    tiny_dataset = [list(range(7)), list(range(7, 10))]
    print("dataset", tiny_dataset)
    for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
        print("中心词：", center, "背景词：", context)
        pass

    # 设最大背景窗口大小为 5，提取二次采样后的数据集中所有的中心词及其背景词
    all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

    # 10.3.2 负采样
    def get_negatives(all_contexts, sampling_weights, K):
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for contexts in all_contexts:
            negatives = []
            while len(negatives) < len(contexts) * K:
                if i == len(neg_candidates):
                    # 根据每个词的权重（sampling_weights），随机生成 $k$ 个词的索引作为噪声词
                    # 为了提高计算效率，可以将 $k$ 的值设置大一点
                    i, neg_candidates = 0, random.choices(population, sampling_weights, k=int(1e5))
                    pass
                neg, i = neg_candidates[i], i + 1
                if neg not in set(contexts):
                    # 如果噪声词不属于背景词就加入噪声词序列
                    negatives.append(neg)
                    pass
                pass
            all_negatives.append(negatives)
            pass
        return all_negatives

    # 噪声词中采样频率 $P(w)$ 设为 $w$ 词频与总词频之比的 $0.75^2$
    sampling_weights = [counter[w] ** 0.75 for w in idx_to_token]
    # 随机采样 $K$ 个噪声词
    all_negatives = get_negatives(all_contexts, sampling_weights, K=5)

    def batchify(data):
        # 小批量读取函数：将中心词（center）、背景词（context）、噪声词（negative）整合在一起
        # 数据中中心词对应的背景词和噪声词最长的链的长度
        max_len = max(len(c) + len(n) for _, c, n in data)
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative in data:
            cur_len = len(context) + len(negative)
            # 中心词数据链
            centers += [center]
            # 中心词对应的「背景词+噪声词」链，链条长度需要统一，不够长的用0补齐
            contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
            # 掩码数据链，用于屏蔽 contexts_negatives 增加的 0 数据
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            # 为背景词和噪声词设置不同的标签
            labels += [[1] * len(context) + [0] * (max_len - len(context))]
            pass
        return nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives), nd.array(masks), nd.array(labels)

    batch_size = 512
    num_workers = 0 if sys.platform.startswith('win32') else 4
    dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True, batchify_fn=batchify, num_workers=num_workers)
    for batch in data_iter:
        for name, data in zip(['centers', 'contexts_negatives', 'masks', 'labels'], batch):
            print(name, 'shape:', data.shape)  # , "内容：", data)
            pass
        break
        pass

    # 10.3.4 跳字模型
    # 1. 嵌入层
    # 词典大小=input_dim, 词向量的维度=output_dim
    embed = nn.Embedding(input_dim=20, output_dim=4)
    embed.initialize()
    print("embed.weight=", embed.weight)
    x = nd.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    print(embed(x))

    # 2. 小批量乘法
    # 输入形状为 $(n,a,b)$ 和 $(n,b,c)$，小批量乘法的输出为 $(n,a,c)$
    X = nd.ones((2, 1, 4))
    Y = nd.ones((2, 4, 5))
    print(nd.batch_dot(X, Y).shape)

    # 3. 跳字模型前向计算
    def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
        # 中心词索引 center 的形状为（批量大小，1）
        # 背景词+噪声词的索引 contexts_and_negatives 的形状为（批量大小，max_len）
        # 小批量乘法输出的形状 （批量大小，1，max_len）
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        pred = nd.batch_dot(v, u.swapaxes(1, 2))
        return pred

    # 10.3.5 训练模型
    # 1. 二元交叉熵损失函数
    loss = gloss.SigmoidBinaryCrossEntropyLoss()
    pred = nd.array([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
    label = nd.array([[1, 0, 0, 0], [1, 1, 0, 0]])
    mask = nd.array([[1, 1, 1, 1], [1, 1, 1, 0]])
    # 掩码为 0 时，会避免参加损失函数的计算
    print("loss=", loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))

    def sigmd(x):
        return -math.log(1 / (1 + math.exp(-x)))

    # 观察系统利用掩码屏蔽损失函数计算的效果
    print("%.7f" % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-1)) / 4))
    print("%.7f" % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))

    # 2. 初始化模型参数
    embed_size = 100
    net = nn.Sequential()
    # 分别构造中心词和背景词的嵌入层
    net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
            nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))

    # 3. 定义训练函数
    def train(net, lr, num_epochs):
        ctx = d2l.try_gpu()
        net.initialize(ctx=ctx, force_reinit=True)
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
        for epoch in range(num_epochs):
            start, l_sum, n = time.time(), 0.0, 0
            for batch in data_iter:
                center, context_negative, mask, label = [data.as_in_context(ctx) for data in batch]
                with autograd.record():
                    pred = skip_gram(center, context_negative, net[0], net[1])
                    l = (loss(pred.reshape(label.shape), label, mask) * mask.shape[1] / mask.sum(axis=1))
                    pass
                l.backward()
                trainer.step(batch_size)
                l_sum += l.sum().asscalar()
                n += l.size
                pass
            print("epoch %d, loss %.2f,time %.2fs" % (epoch + 1, l_sum / n, time.time() - start))
            pass
        pass

    train(net, 0.005, 5)

    def get_similar_tokens(query_token, k, embed):
        W = embed.weight.data()
        x = W[token_to_idx[query_token]]
        cos = nd.dot(W, x) / (nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
        topk = nd.topk(cos, k=k + 1, ret_typ='indices').asnumpy().astype('int32')
        for i in topk[1:]:
            print("cosine sim=%.3f: %s" % (cos[i].asscalar(), (idx_to_token[i])))
            pass
        pass

    get_similar_tokens('chip', 3, net[0])

    pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
