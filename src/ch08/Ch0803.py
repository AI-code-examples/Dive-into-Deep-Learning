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
@Desc       :   Sec
@小结：
"""
import random

from d2l import mxnet as d2l
from mxnet import nd, np, npx, autograd
from mxnet.gluon import nn
from tools import *

npx.set_np()


# ----------------------------------------------------------------------
def main():
    # 8.3.3 Natural Language Statistics
    tokens = d2l.tokenize(d2l.read_time_machine())

    show_subtitle("corpus")
    corpus = [token for line in tokens for token in line]
    print(corpus[:10])

    show_subtitle("vocab")
    vocab = d2l.Vocab(corpus)
    print(vocab.token_freqs[:10])

    import matplotlib.pyplot as plt
    freqs = [x for x in range(1000, 1, -1)]
    d2l.plot(freqs, xlabel='x', ylabel='x', xscale='log', yscale='log')

    plt.figure()
    freqs = [freq for _, freq in vocab.token_freqs]
    d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')

    show_subtitle("bigram tokens")
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    print(bigram_tokens[:10])

    show_subtitle("bigram vocab")
    bigram_vocab = d2l.Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

    show_subtitle("trigram")
    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = d2l.Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

    bigram_freqs = [freq for _, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for _, freq in trigram_vocab.token_freqs]
    plt.figure()
    d2l.plot([freqs, bigram_freqs, trigram_freqs], legend=['unigram', 'bigram', 'trigram'],
             xlabel='x', ylabel='x', xscale='log', yscale='log')

    # 8.3.4 Reading Long Sequence Data
    my_seq = list(range(35))
    # 8.3.4.1 Random Sampling
    show_subtitle("Random Sampling")
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print("X: ", X)
        print("Y: ", Y)

    # 8.3.4.2 Sequential Partitioning
    show_subtitle("Sequential Partitioning")
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print("X: ", X)
        print("Y: ", Y)
    pass


# ----------------------------------------------------------------------
class SeqDataLoader:
    """An iiterator to load sequence data."""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using sequential partitioning."""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = np.array(corpus[offset:offset + num_tokens])
    Ys = np.array(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


def seq_data_iter_random(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using random sampling"""
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos + num_steps]

    num_batchs = num_subseqs // batch_size
    for i in range(0, batch_size * num_batchs, batch_size):
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield np.array(X), np.array(Y)


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
