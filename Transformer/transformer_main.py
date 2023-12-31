'''
Transformer - TensorFlow
'''

import os
import random
import numpy as np
import tensorflow as tf
import transformer_models as models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from Prepare_Data import Vocab
from Prepare_Data import DataLoader

if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    '''
    1. データの準備
    '''
    #Vocabを使い辞書を構築
    data_dir_en = os.path.join(os.path.dirname(__file__), 'Data')
    data_dir_ja = data_dir_en = os.path.join(os.path.dirname(__file__), 'Data')

    en_train_path = os.path.join(data_dir_en, 'train.en')
    en_val_path = os.path.join(data_dir_en, 'dev.en')
    en_test_path = os.path.join(data_dir_en, 'test.en')

    ja_train_path = os.path.join(data_dir_ja, 'train.ja')
    ja_val_path = os.path.join(data_dir_ja, 'dev.ja')
    ja_test_path = os.path.join(data_dir_ja, 'test.ja')

    en_vocab = Vocab()
    ja_vocab = Vocab()

    en_vocab.fit(en_train_path)
    ja_vocab.fit(ja_train_path)

    #単語列をID列に変換
    #入力系列 - BOS/EOSは不要
    x_train = en_vocab.transform(en_train_path)
    x_val = en_vocab.transform(en_val_path)
    x_test = en_vocab.transform(en_test_path)
    #出力系列 - BOS/EOSが必要なため事前に付与
    t_train = ja_vocab.transform(ja_train_path, bos=True, eos=True)
    t_val = ja_vocab.transform(ja_val_path, bos=True, eos=True)
    t_test = ja_vocab.transform(ja_test_path, bos=True, eos=True)


    #長さが同程度のデータを集めミニバッチを構築
    def sort(x, t):
        lens = [len(i) for i in x]
        indices = sorted(range(len(lens)), key=lambda i: -lens[i]) #降順
        x = [x[i] for i in indices]
        t = [t[i] for i in indices]

        return (x, t)

    #系列長に沿ってソート
    (x_train, t_train) = sort(x_train, t_train)
    (x_val, t_val) = sort(x_val, t_val)
    (x_test, t_test) = sort(x_test, t_test)

    #DataLoaderを使いミニバッチに分ける
    train_dataloader = DataLoader((x_train, t_train))
    val_dataloader = DataLoader((x_val, t_val))
    test_dataloader = DataLoader((x_test, t_test), batch_size=1)


    '''
    2. モデルの構築
    '''
    depth_x = len(en_vocab.i2w)
    depth_t = len(ja_vocab.i2w)

    model = models.Transformer(depth_x, depth_t, N=3, h=4, d_model=128, d_ff=256, maxlen=20)


    '''
    3. モデルの学習・評価
    '''
    #事前設定
    criterion = tf.losses.CategoricalCrossentropy()
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    train_loss = metrics.Mean()
    val_loss = metrics.Mean()

    def compute_loss(t, y):
        return criterion(t, y)

    def train_step(x, t, depth_t):
        with tf.GradientTape() as tape:
            preds = model(x, t)
            t = t[:, 1:] #出力はBOS以降となるのでBOS以降で比較
            mask_t = tf.cast(tf.not_equal(t, 0), tf.float32) #出力系列用のマスク計算
            t = tf.one_hot(t, depth=depth_t, dtype=tf.float32) #出力系列をone-hotベクトルに
            t = t * mask_t[:, :, tf.newaxis] #出力系列にマスクを適応
            loss = compute_loss(t, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

        return preds

    def val_step(x, t, depth_t):
        preds = model(x, t)
        t = t[:, 1:] #出力はBOS以降となるのでBOS以降で比較
        #モデルの出力系列内のパディング部分に対してマスク処理
        mask_t = tf.cast(tf.not_equal(t, 0), tf.float32)
        t = tf.one_hot(t, depth=depth_t, dtype=tf.float32)
        t = t * mask_t[:, :, tf.newaxis]
        loss = compute_loss(t, preds)
        val_loss(loss)

        return preds

    #テストデータに対しては誤差の値は求めず翻訳文の出力のみ行う
    def test_step(x):
        preds = model(x)
        return preds


    #学習開始
    epochs = 30

    for epoch in range(epochs):
        print('-' * 20)
        print('epoch: {}'.format(epoch+1))

        for (x, t) in train_dataloader:
            train_step(x, t, depth_t)

        for (x, t) in val_dataloader:
            val_step(x, t, depth_t)

        print('loss: {:.3f}, val_loss: {:.3}'.format(
            train_loss.result(),
            val_loss.result()
        ))

        for idx, (x, t) in enumerate(test_dataloader):
            preds = test_step(x)

            #decode()のためにnumpy配列に変換
            source = x.numpy().reshape(-1)
            target = t.numpy().reshape(-1)
            out = preds.numpy().reshape(-1)

            #ID列を単語列に変換
            source = ' '.join(en_vocab.decode(source))
            target = ' '.join(ja_vocab.decode(target))
            out = ' '.join(ja_vocab.decode(out))

            print('>', source)
            print('=', target)
            print('<', out)
            print()

            if idx >= 9:
                break
