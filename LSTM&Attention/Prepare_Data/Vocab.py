'''
Vocab
辞書の構築と変換のためのファイル
(辞書 - コープス内の単語(トークン)にIDを割り振った対応表)
'''

class Vocab(object):
    def __init__(self):
        self.w2i = {} #Word to ID　の辞書
        self.i2w = {} #ID to Work　の辞書
        self.special_chars = ['<pad>', '<s>', '</s>', '<unk>'] #pad = padding, s = 文章の開始, /s =　文章の終了、unk = unknown 
        self.bos_char = self.special_chars[1]
        self.eos_char = self.special_chars[2]
        self.oov_char = self.special_chars[3]

    #ファイルを読み込み、辞書を構築する処理
    #(ファイルを一行ずつ読み込み、初見の単語が出たら新しく辞書に追加する処理)
    #作るたび単語とIDの対応は変わるため、別ファイルで読み込み際は保存する処理を加える
    def fit(self, path):
        self._words = set()

        #ファイル内の各文をリスト化
        with open(path, 'r') as f:
            sentences = f.read().splitlines()

        #リスト内をtraversalし、新出の単語を保持
        for sentence in sentences:
            self._words.update(sentence.split())

        #予約語のID分ずらす
        self.w2i = {w: (i + len(self.special_chars))
                    for i, w in enumerate(self._words)}

        #予約語を辞書に保存
        for i, w in enumerate(self.special_chars):
            self.w2i[w] = i

        self.i2w = {i: w for w, i in self.w2i.items()}

    #ファイルを読み込み全体をID列に変換する処理
    #fit()により構築した辞書を用いて、単語をIDに変換していく
    def transform(self, path, bos=False, eos=False):
        output = []

        #ファイル内の各文をリスト化
        with open(path, 'r') as f:
            sentences = f.read().splitlines()

        for sentence in sentences:
            sentence = sentence.split()
            #文章の最初の場合<s>を文頭につける
            if bos:
                sentence = [self.bos_char] + sentence
            #文末の場合</s>を文末につける
            if eos:
                sentence = sentence + [self.eos_char]
            output.append(self.encode(sentence)) #ID列に変換

        return output

    #実際にID列に変換処理を行う関数
    def encode(self, sentence):
        output = []

        for w in sentence:
            #辞書にない単語は未知語(unk)に変換
            if w not in self.w2i:
                idx = self.w2i[self.oov_char]
            else:
                idx = self.w2i[w]
            output.append(idx)

        return output

    #ID列を単語列に戻す処理
    def decode(self, sentence):
        return [self.i2w[id] for id in sentence]
