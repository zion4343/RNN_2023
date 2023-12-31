'''
Scaled Dot-Product Attention
query, key, valueの3つの側面で考える
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class ScaledDotProductAttention(Layer):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k #dimention of key
        self.scaler = np.sqrt(d_k)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, q, k, v, mask=None): #q = query, k = key, v = value, mask = マスク処理をする場合マスクを受け取る
        score = tf.einsum('ijk,ilk->ijl', q, k) / self.scaler #スコア関数の計算
        score = score - tf.reduce_max(score, axis=-1, keepdims=True) #加重平均の計算
        score = tf.exp(score)
        
        #マスク処理
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, tf.newaxis, :]
            mask = tf.cast(mask, tf.float32)
            score = score * mask

        a = score / tf.reduce_sum(score, axis=-1, keepdims=True) #加重平均の計算
        c = tf.einsum('ijk,ikl->ijl', a, v)

        return c

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)

    def compute_mask(self, inputs, mask):
        return mask
