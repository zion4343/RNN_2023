'''
Data Loader
文章データをミニバッチに分ける関数
'''

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataLoader(object):
    def __init__(self, dataset,
                 batch_size=100,
                 shuffle=False,
                 random_state=None):
        self.dataset = list(zip(dataset[0], dataset[1]))
        self.batch_size = batch_size
        self.shuffle = shuffle

        if random_state is None:
            random_state = np.random.RandomState(123) #乱数固定

        self.random_state = random_state
        self._idx = 0 #何番目のデータを処理しているかを表すindex
        self._reset() #データのシャッフル & データ番号の初期化

    #len()により得られる値を定義する処理
    def __len__(self):
        N = len(self.dataset)
        b = self.batch_size
        return N // b + bool(N % b)

    #iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration() #例外処理

        x, t = zip(*self.dataset[self._idx:(self._idx + self.batch_size)])
        x = pad_sequences(x, padding='post') #データのパディング
        t = pad_sequences(t, padding='post') #データのパディング

        #データをTensorflowのテンソル型に変換
        x = tf.convert_to_tensor(x, dtype=tf.int32)
        t = tf.convert_to_tensor(t, dtype=tf.int32)

        self._idx += self.batch_size

        return x, t

    def _reset(self):
        if self.shuffle:
            self.dataset = shuffle(self.dataset,
                                   random_state=self.random_state)
        self._idx = 0
