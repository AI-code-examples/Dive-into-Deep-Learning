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
@Desc       :   Sec 10.12 机器翻译
@小结：
1.  可以将编码器-解码器和注意力机制应用于机器翻译中
2.  BLEU 可以用来评价翻译结果
"""
import collections
import d2lzh as d2l
import data
import io
import math
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    # 10.12.2 包含注意力机制的编码器-解码器
    # 1. 编码器
    # 创建一个批量大小为 4， 时间步数为 7 的小批量序列输入
    # 创建门控循环单元的隐藏层个数为 2，隐藏单元个数为 16
    # 编码器对该输入执行前向计算后返回的输出形状为（时间步数，批量大小，隐藏单元个数）
    # 门控循环单元在最终时间步的多层隐藏状态的形状为（隐藏层个数，批量大小，隐藏单元个数）
    encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.initialize()
    output, state = encoder(nd.zeros((4, 7)), encoder.begin_state(batch_size=4))
    print(output.shape, state[0].shape)

    # 2. 注意力机制
    # Dense() 的 flatten 选项的效果
    dense = nn.Dense(2, flatten=False)
    dense.initialize()
    print(dense(nd.zeros((3, 5, 7))).shape)

    seq_len, batch_size, num_hiddens = 10, 4, 8
    model = attention_model(10)
    model.initialize()
    enc_states = nd.zeros((seq_len, batch_size, num_hiddens))
    dec_states = nd.zeros((batch_size, num_hiddens))
    print(attention_forward(model, enc_states, dec_states).shape)

    embed_size, num_hiddens, num_layers = 64, 64, 2
    attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
    encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers, drop_prob)
    decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers, attention_size, drop_prob)
    train(encoder, decoder, dataset, lr, batch_size, num_epochs)

    intput_seq = 'ils regardent .'
    print(translate(encoder, decoder, intput_seq, max_seq_len))

    def score(input_seq, label_seq, k):
        pred_tokens = translate(encoder, decoder, input_seq, max_seq_len)
        label_tokens = label_seq.split(' ')
        print("bleu %.3f, predict: %s" % (bleu(pred_tokens, label_tokens, k), ' '.join(pred_tokens)))
        pass

    score('ils regardent .', 'they are watching .', k=2)
    score('ils sont canadiens .', 'they are canadian .', k=2)

    pass


class Encoder(nn.Block):
    # 1. 编码器
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        pass

    def forward(self, inputs, state):
        # 输入形状是（批量大小，时间步数）。将输出的样本维和时间步维交换。
        embedding = self.embedding(inputs).swapaxes(0, 1)
        return self.rnn(embedding, state)

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class Decoder(nn.Block):
    # 3. 包含注意力机制的解码器
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, attention_size, drop_prob=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(attention_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Dense(vocab_size, flatten=False)
        pass

    def forward(self, cur_input, state, enc_states):
        # 使用注意力机制计算背景向量
        c = attention_forward(self.attention, enc_states, state[0][-1])
        # 将嵌入后的输入向量和背景向量在特征维连接
        input_and_c = nd.concat(self.embedding(cur_input), c, dim=1)
        # 为输入向量和背景向量的连结增加时间步维，时间步个数为 1
        output, state = self.rnn(input_and_c.expand_dims(0), state)
        # 移除时间步维，输出形状为（批量大小，输出词典大小）
        output = self.out(output).squeeze(axis=0)
        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state


def bleu(pred_tokens, label_tokens, k):
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i + n])] += 1
            pass
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i:i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i + n])] -= 1
                pass
            pass
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
        pass
    return score


def translate(encoder: Encoder, decoder: Decoder, input_seq: str, max_seq_len: int) -> object:
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = nd.array([in_vocab.to_indices(in_tokens)])
    enc_state = encoder.begin_state(batch_size=1)
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = nd.array([out_vocab.token_to_idx[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(axis=1)
        pred_token = out_vocab.idx_to_token[int(pred.asscalar())]
        if pred_token == EOS:  # 当任一时间步搜索出 EOS 时，输出序列即完成
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
            pass
        pass
    return output_tokens


def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    encoder.initialize(init.Xavier(), force_reinit=True)
    decoder.initialize(init.Xavier(), force_reinit=True)
    enc_trainer = gluon.Trainer(encoder.collect_params(), 'adam', {'learning_rate': lr})
    dec_trainer = gluon.Trainer(decoder.collect_params(), 'adam', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.9
        for X, Y in data_iter:
            with autograd.record():
                l = batch_loss(encoder, decoder, X, Y, loss)
                pass
            l.backward()
            enc_trainer.step(1)
            dec_trainer.step(1)
            l_sum += l.asscalar()
            pass
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
            pass
        pass
    pass


def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state(batch_size=batch_size)
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始化解码器的隐藏状态
    dec_state = decoder.begin_state(enc_state)
    # 解码器在最初时间步的输入是 BOS
    dec_input = nd.array([out_vocab.token_to_idx[BOS]] * batch_size)
    # 使用掩码变量 mask 来忽略掉标签为填充项 PAD 的损失
    mask, num_not_pad_tokens = nd.ones(shape=(batch_size,)), 0
    l = nd.array([0])
    for y in Y.T:
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # 使用强制教学
        num_not_pad_tokens += mask.sum().asscalar()
        # 当遇到 EOS 时，序列后面的词将均为 PAD，相应位置的掩码设成 0
        mask = mask * (y != out_vocab.token_to_idx[EOS])
        pass
    return l / num_not_pad_tokens


def attention_model(attention_size):
    # 注意力模型
    # attention_size 是超参数
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation='tanh', use_bias=False, flatten=False),
              nn.Dense(1, use_bias=False, flatten=False))
    return model


def attention_forward(model, enc_states, dec_states):
    # 注意力模型的前向计算
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
    dec_states = nd.broadcast_axis(dec_states.expand_dims(0), axis=0, size=enc_states.shape[0])
    enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)
    e = model(enc_and_dec_states)  # 形状为（时间步数，批量大小，1）
    alpha = nd.softmax(e, axis=0)  # 在时间步维度做 Softmax 运算
    return (alpha * enc_states).sum(axis=0)  # 返回背景变量


def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    # 先将序列中所有的词都记录在 all_tokens 中以便之后构造词典
    # 然后在该序列后面添加 PAD 直到序列的长度达到 max_seq_len
    # 最后将序列保存在 all_seqs
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)


def build_data(all_tokens, all_seqs):
    # 使用所有的词来构造词典，并且将所有序列中的词变换为词索引后构造 NDArray 实例
    vocab = text.vocab.Vocabulary(collections.Counter(all_tokens), reserved_tokens=[PAD, BOS, EOS])
    indices = [vocab.to_indices(seq) for seq in all_seqs]
    return vocab, nd.array(indices)


def read_data(max_seq_len):
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('../data/fr-en-small.txt') as f:
        lines = f.readlines()
        pass
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue    # 如果加上 EOS 后长于 max_seq_len，则忽略掉这个样本
            pass
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
        pass
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, gdata.ArrayDataset(in_data, out_data)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    # <pad>(padding) 符号用来添加在较短序列的后面，直到所有序列等长
    # <bos> 符号表示序列的开始
    # <eos> 符号表示序列的结束
    PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
    max_seq_len = 7
    in_vocab, out_vocab, dataset = read_data(max_seq_len)
    # print(dataset[0])

    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
