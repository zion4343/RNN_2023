'''
Positional Encoding
LSTMやGRUなどの再起的な処理の代わりに、系列データに直接順序・位置関係の情報を埋め込む層
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class PositionalEncoding(Layer):
    def __init__(self, output_dim,
                 maxlen=6000):
        super().__init__()
        self.output_dim = output_dim
        self.maxlen = maxlen

    def build(self, input_shape):
        self.PE = self.add_weight(name='PE',
                                  shape=(self.maxlen, self.output_dim),
                                  initializer=self.initializer,
                                  trainable=False, #学習しないよう設定
                                  dtype=tf.float32)

        super().build(input_shape)

    def call(self, x):
        pe = self.PE[tf.newaxis, :tf.shape(x)[1], :]
        return x + pe

    #行列PEの初期値設定
    def initializer(self, input_shape, dtype=tf.float32):
        pe = np.array([[pos / np.power(10000, 2 * (i // 2) / self.output_dim)
                       for i in range(self.output_dim)]
                       for pos in range(self.maxlen)])

        pe[:, 0::2] = np.sin(pe[:, 0::2]) #2i
        pe[:, 1::2] = np.cos(pe[:, 1::2]) #2i+1

        return tf.convert_to_tensor(pe, dtype=tf.float32)
