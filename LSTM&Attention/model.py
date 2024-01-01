'''
EncoderDecorder - Tensorflow
モデルを構成するクラスの定義
- EncoderDecorder
- Encoder
- Decorder
'''

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
from Layers import Attention

#EncorderDecorderクラス
#構成「
class EncoderDecoder(Model):
    def __init__(self, input_dim, hidden_dim, output_dim, maxlen=20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

        self.maxlen = maxlen
        self.output_dim = output_dim

    def call(self, source, target=None, use_teacher_forcing=False):
        batch_size = source.shape[0]
        if target is not None:
            len_target_sequences = target.shape[1]
        else:
            len_target_sequences = self.maxlen

        hs, states = self.encoder(source)

        y = tf.ones((batch_size, 1), dtype=tf.int32)
        output = tf.zeros((batch_size, 1, self.output_dim), dtype=tf.float32)

        for t in range(len_target_sequences):
            out, states = self.decoder(y, hs, states, source=source)
            output = tf.concat([output, out[:, :1]], axis=1)

            if use_teacher_forcing and target is not None:
                y = target[:, t][:, tf.newaxis]
            else:
                y = tf.argmax(out, axis=-1, output_type=tf.int32)

        return output[:, 1:]

#Encorderクラス
#構成 「埋め込み層 - LSTM層」
class Encoder(Model):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.embedding = Embedding(input_dim, hidden_dim, mask_zero=True)
        self.lstm = LSTM(hidden_dim, activation='tanh',
                         recurrent_activation='sigmoid',
                         kernel_initializer='glorot_normal',
                         recurrent_initializer='orthogonal',
                         return_state=True,
                         return_sequences=True) #層の出力をシーケンスとするか否か、出力hが時間軸を持つ値となる

    def call(self, x):
        x = self.embedding(x)
        h, state_h, state_c = self.lstm(x)

        return h, (state_h, state_c)

#Decorderクラス
#構成 「埋め込み層 - LSTM層 - Attention層 - 出力層」
class Decoder(Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.embedding = Embedding(output_dim, hidden_dim)
        self.lstm = LSTM(hidden_dim, activation='tanh',
                         recurrent_activation='sigmoid',
                         kernel_initializer='glorot_normal',
                         recurrent_initializer='orthogonal',
                         return_state=True,
                         return_sequences=True)
        self.attn = Attention(hidden_dim, hidden_dim)
        self.out = Dense(output_dim, kernel_initializer='glorot_normal', activation='softmax')

    def call(self, x, hs, states, source=None):
        x = self.embedding(x)
        ht, state_h, state_c = self.lstm(x, states)
        ht = self.attn(ht, hs, source=source)
        y = self.out(ht)

        return y, (state_h, state_c)