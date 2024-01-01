'''
Attention - TensorFlow
各時刻によって動的に変わるベクトルを含むモデル
重要な情報に注目できるようにする層
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer


#Attention層
#h_s及びh_tを受け取り~h_tを出力する層
class Attention(Layer):
    def __init__(self, output_dim, hidden_dim):
        super().__init__()
        self.output_dim = output_dim #~h_tの次元
        self.hidden_dim = hidden_dim #h_sとh_tの次元

    #パラメータの定義
    def build(self, input_shape):
        self.W_a = self.add_weight(name='W_a',
                                   shape=(self.hidden_dim,
                                          self.hidden_dim),
                                   initializer='glorot_normal',
                                   trainable=True)

        self.W_c = self.add_weight(name='W_c',
                                   shape=(self.hidden_dim + self.hidden_dim,
                                          self.output_dim),
                                   initializer='glorot_normal',
                                   trainable=True)

        self.b = self.add_weight(name='b',
                                 shape=(self.output_dim),
                                 initializer='zeros',
                                 trainable=True)

        super().build(input_shape)

    #順伝道の処理の定義
    def call(self, ht, hs, source=None):
        #スコア関数の計算
        score = tf.einsum('ijk,kl->ijl', hs, self.W_a)
        score = tf.einsum('ijk,ilk->ijl', ht, score)

        #加重平均の計算(スコア関数のソフトマックス計算)
        score = score - tf.reduce_max(score, axis=-1, keepdims=True)
        score = tf.exp(score)
        
        #マスク処理
        if source is not None:
            #入力データの系列長の取得
            len_source_sequences = tf.reduce_sum(tf.cast(tf.not_equal(source, 0), tf.int32), axis=1)
            #系列長を引数としてマスクを生成
            mask_source = tf.cast(tf.sequence_mask(len_source_sequences, tf.shape(score)[-1]), tf.float32)
            #パディング部分はAttentionの重みがゼロになるように計算
            score = score * mask_source[:, tf.newaxis, :]
        
        a = score / tf.reduce_sum(score, axis=-1, keepdims=True) #加重平均の計算
        c = tf.einsum('ijk,ikl->ijl', a, hs) #文脈ベクトルの計算
        
        #出力の計算
        h = tf.concat([c, ht], -1)
        return tf.nn.tanh(tf.einsum('ijk,kl->ijl', h, self.W_c) + self.b)
