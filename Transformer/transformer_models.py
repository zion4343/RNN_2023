'''
Transformerのモデルを構築するファイル
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from Layers import PositionalEncoding
from Layers import LayerNormalization
from Layers import MultiHeadAttention


#Transformerクラス
class Transformer(Model):
    def __init__(self, depth_source, depth_target, N=6, h=8, d_model=512, d_ff=2048, p_dropout=0.1, maxlen=128):
        super().__init__()
        self.encoder = Encoder(depth_source, N=N, h=h, d_model=d_model, d_ff=d_ff, p_dropout=p_dropout, maxlen=maxlen)
        
        self.decoder = Decoder(depth_target, N=N, h=h, d_model=d_model, d_ff=d_ff, p_dropout=p_dropout, maxlen=maxlen)
        
        self.out = Dense(depth_target, activation='softmax')
        
        self.maxlen = maxlen


    def call(self, source, target=None):
        mask_source = self.sequence_mask(source) #Source系列のマスク

        hs = self.encoder(source, mask=mask_source)

        #訓練時の順伝道の処理 (Targetも教師データがある場合)
        #Attentionを用いることで再起計算は不要
        if target is not None:
            target = target[:, :-1] #デコーダーへの入力ではEOSは不要
            len_target_sequences = target.shape[1]
            mask_target = self.sequence_mask(target)[:, tf.newaxis, :] #Target系列のパディング用マスク
            subsequent_mask = self.subsequence_mask(target) #Target系列の未来情報用マスク
            mask_target = tf.greater(mask_target + subsequent_mask, 1) #2つのマスクのうちどちらがが0であれば、その部分をマスクする

            y = self.decoder(target, hs,
                             mask=mask_target,
                             mask_source=mask_source)
            output = self.out(y)
            
        #テスト時の順伝道の処理 (Targetに教師データがない場合)
        #時刻ごとに再起計算が必要    
        else:
            batch_size = source.shape[0]
            len_target_sequences = self.maxlen

            output = tf.ones((batch_size, 1), dtype=tf.int32) #BOSを生成

            #時刻ごとに処理する再起計算
            for t in range(len_target_sequences - 1):
                mask_target = self.subsequence_mask(output)
                out = self.decoder(output, hs,
                                   mask=mask_target,
                                   mask_source=mask_source)
                out = self.out(out)[:, -1:, :] #出力の最終時刻のデータを付け足す
                out = tf.argmax(out, axis=-1, output_type=tf.int32)
                output = tf.concat([output, out], axis=-1)

        return output


    #パディング処理向けのマスク
    def sequence_mask(self, x):
        len_sequences = \
            tf.reduce_sum(tf.cast(tf.not_equal(x, 0),
                                  tf.int32), axis=1)
        mask = \
            tf.cast(tf.sequence_mask(len_sequences,
                                     tf.shape(x)[-1]), tf.float32)
        return mask


    #デコーダーのself attentionにて時系列データの各時刻から未来の情報が見えないようにするマスク
    #未来の情報部分を0とするマスク
    def subsequence_mask(self, x):
        shape = (x.shape[1], x.shape[1])
        mask = np.tril(np.ones(shape, dtype=np.int32), k=0)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        return tf.tile(mask[tf.newaxis, :, :], [x.shape[0], 1, 1])


#Encorderクラス
#構成 - 「埋め込み層 - Positional Encording - 複数のEncorderLayer」 
class Encoder(Model):
    def __init__(self, depth_source, N=6, h=8, d_model=512, d_ff=2048, p_dropout=0.1, maxlen=128):
        super().__init__()
        self.embedding = Embedding(depth_source, d_model, mask_zero=True)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.encoder_layers = [
            EncoderLayer(h=h, d_model=d_model, d_ff=d_ff, p_dropout=p_dropout, maxlen=maxlen) 
            for _ in range(N)
        ]

    def call(self, x, mask=None):
        x = self.embedding(x)
        y = self.pe(x)
        for encoder_layer in self.encoder_layers:
            y = encoder_layer(y, mask=mask)

        return y

#EncorderLayerクラス
#構成 - 「Multi-Head Attention - Add&Norm - Feed Forward - Add&Norm」 
class EncoderLayer(Model):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        self.attn = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(p_dropout)
        self.norm1 = LayerNormalization()
        self.ff = FFN(d_model, d_ff)
        self.dropout2 = Dropout(p_dropout)
        self.norm2 = LayerNormalization()

    def call(self, x, mask=None):
        h = self.attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h) #x + hがAddの役割

        y = self.ff(h)
        y = self.dropout2(y)
        y = self.norm2(h + y) #h + yがAddの役割

        return y


#Decorderクラス
#構成 - 「埋め込み層 - Positional Encording - 複数のDecorderLayer」 
class Decoder(Model):
    def __init__(self,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        self.embedding = Embedding(depth_target, d_model, mask_zero=True)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.decoder_layers = [
            DecoderLayer(h=h, d_model=d_model, d_ff=d_ff, p_dropout=p_dropout, maxlen=maxlen) 
            for _ in range(N)
        ]

    def call(self, x, hs, mask=None, mask_source=None):
        x = self.embedding(x)
        y = self.pe(x)

        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, hs, mask=mask, mask_source=mask_source)

        return y

#DecorderLayerクラス
#構成 - 「Multi-Head Attention - Add&Norm - Multi-Head Attention - Add&Norm - Feed Forward - Add&Norm」 
class DecoderLayer(Model):
    def __init__(self, h=8, d_model=512, d_ff=2048, p_dropout=0.1, maxlen=128):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(p_dropout)
        self.norm1 = LayerNormalization()
        self.src_tgt_attn = MultiHeadAttention(h, d_model)
        self.dropout2 = Dropout(p_dropout)
        self.norm2 = LayerNormalization()
        self.ff = FFN(d_model, d_ff)
        self.dropout3 = Dropout(p_dropout)
        self.norm3 = LayerNormalization()

    def call(self, x, hs, mask=None, mask_source=None):
        h = self.self_attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)

        z = self.src_tgt_attn(h, hs, hs,
                              mask=mask_source)
        z = self.dropout2(z)
        z = self.norm2(h + z)

        y = self.ff(z)
        y = self.dropout3(y)
        y = self.norm3(z + y)

        return y


#Feed Forward Neural Network クラス
#構成 - 「隠れ層 - 出力層」
class FFN(Model):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.l1 = Dense(d_ff, activation='relu')
        self.l2 = Dense(d_model, activation='linear')

    def call(self, x):
        h = self.l1(x)
        y = self.l2(h)
        return y