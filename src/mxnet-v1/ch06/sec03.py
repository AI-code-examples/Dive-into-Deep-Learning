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
@Desc       :   Sec 6.3语言模型数据集（歌词）
@小结：
1.  时序数据采样方式包括随机采样和相邻采样，使用这两种方式的循环神经网络训练在实现上略有不同
"""
import d2lzh as d2l
import mxnet as mx
import random
import zipfile
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures, get_root_path


# ----------------------------------------------------------------------
def main():
    with zipfile.ZipFile(get_root_path() + "data/jaychou_lyrics.txt.zip") as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
            pass
        pass
    print(corpus_chars[:40])
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    # 字符串转换成不重复的 set，再转换成 list，再转换成字典 (字，索引）
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    print(vocab_size)
    # 将字符转换成数字
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    sample = corpus_indices[:20]
    print("chars:", ''.join([idx_to_char[idx] for idx in sample]))
    print("indices:", sample)

    my_seq = list(range(30))
    show_subtitle("1. 随机采样")
    for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
        print("X:", X, "\n", "Y:", Y)
        pass

    show_subtitle("2. 相邻采样")
    for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
        print("X:", X, "\n", "Y:", Y)
        pass
    pass


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """
    随机采样函数
    """
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos:pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i:i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """
    相邻采样函数
    """
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0:batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i:i + num_steps]
        Y = indices[:, i + 1:i + num_steps + 1]
        yield X, Y


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
