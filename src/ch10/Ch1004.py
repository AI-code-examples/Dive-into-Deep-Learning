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
from mxnet.gluon import nn, rnn
from tools import beep_end, show_subtitle, show_title, show_figures

npx.set_np()


# ----------------------------------------------------------------------
def main():
    encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.initialize()
    decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder.initialize()
    X = np.zeros((4, 7))  # ('batch_size', 'num_steps')
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', 'i lost .', "he\'s calm .", "i\'m home ."]
    fras = ['va !', "j\'ai perdu .", 'il est calme .', 'je suis che moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, bleu {d2l.bleu(translation, fra, k=2):.3f}')

    attention_weights = np.concatenate([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((1, 1, -1, num_steps))
    # +1 是为也包含序列末端的令牌
    d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1], xlabel='Keys positions', ylabel='Query positions')
    pass


class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface."""

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self._attention_weights = None

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # 'outputs' shape: ('num_steps', 'batch_size', 'num_hiddens')
        # 'hidden_state[0]' shape: ('num_layers', 'batch_size', 'num_hiddens')
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
