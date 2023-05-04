# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/18 下午 03:36

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
import collections
import six


class Vocab:
    """
    根據本地的vocab文件，構造一個詞表
    vocab = Vocab()
    print(vocab.itos)  # 得到一個列表，返回詞表中的每一個詞；
    print(vocab.itos[2])  # 通過索引返回得到詞表中對應的詞；
    print(vocab.stoi)  # 得到一個字典，返回詞表中每個詞的索引；
    print(vocab.stoi['我'])  # 通過單詞返回得到詞表中對應的索引
    print(len(vocab))  # 返回詞表長度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一個列表，返回詞表中的每一個詞；
    print(vocab.itos[2])  # 通過索引返回得到詞表中對應的詞；
    print(vocab.stoi)  # 得到一個字典，返回詞表中每個詞的索引；
    print(vocab.stoi['我'])  # 通過單詞返回得到詞表中對應的索引
    """
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    對一個List中的元素進行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一個維度
        padding_value:
        max_len :
                當max_len = 50時，表示以某個固定長度對樣本進行padding，多餘的截掉；
                當max_len=None是，表示以當前batch中最長樣本的長度對其它進行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    # print("out_tensors:",out_tensors)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


def cache(func):
    """
    本修飾器的作用是將SQuAD數據集中data_process()方法處理後的結果進行緩存，下次使用時可直接載入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"緩存文件 {data_path} 不存在，重新處理並緩存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"緩存文件 {data_path} 存在，直接載入緩存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class LoadSingleSentenceClassificationDataset:
    def __init__(self,
                 vocab_path='./vocab.txt',  #
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True
                 ):

        """

        :param vocab_path: 本地詞表vocab.txt的路徑
        :param tokenizer:
        :param batch_size:
        :param max_sen_len: 在對每個batch進行處理時的配置；
                            當max_sen_len = None時，即以每個batch中最長樣本長度為標準，對其它進行padding
                            當max_sen_len = 'same'時，以整個數據集中最長樣本為標準，對其它進行padding
                            當max_sen_len = 50， 表示以某個固定長度符樣本進行padding，多餘的截掉；
        :param split_sep: 文本和標籤之前的分隔符，默認為'\t'
        :param max_position_embeddings: 指定最大樣本長度，超過這個長度的部分將本截取掉
        :param is_sample_shuffle: 是否打亂訓練集樣本（只針對訓練集）
                在後續構造DataLoader時，驗證集和測試集均指定為了固定順序（即不進行打亂），修改程序時請勿進行打亂
                因為當shuffle為True時，每次通過for循環遍歷data_iter時樣本的順序都不一樣，這會導致在模型預測時
                返回的標籤順序與原始的順序不一樣，不方便處理。

        """
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        # self.UNK_IDX = '[UNK]'

        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle


    def data_process(self, filepath, only_predict=False):
        """
        將每一句話中的每一個詞根據字典轉換成索引的形式，同時返回所有樣本中最長樣本的長度
        :param filepath: 數據集路徑
        :return:
        """
        if only_predict:
            contents = [filepath]
            label_len = len(contents)
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(filepath)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT預訓練模型只取前512個字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(label_len), dtype=torch.long)
            data = []
            data.append((tensor_, l))
            return data
        else:
            raw_iter = open(filepath, encoding="utf-8").readlines()
            data = []
            max_len = 0
            for raw in tqdm(raw_iter, ncols=80):
                line = raw.rstrip("\n").split(self.split_sep)
                s, l = line[0], line[1]
                # print("s:", s)
                # print("l:", l)
                tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(s)]
                if len(tmp) > self.max_position_embeddings - 1:
                    tmp = tmp[:self.max_position_embeddings - 1]  # BERT預訓練模型只取前512個字符
                tmp += [self.SEP_IDX]
                tensor_ = torch.tensor(tmp, dtype=torch.long)
                l = torch.tensor(int(l), dtype=torch.long)
                max_len = max(max_len, tensor_.size(0))
                data.append((tensor_, l))
            return data, max_len

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 pre_content=None,
                                 only_test=False,
                                 only_predict=False):

        if only_predict:
            predict_data = self.data_process(pre_content, only_predict=True)
            predict_iter = DataLoader(predict_data, batch_size=self.batch_size,
                                   shuffle=False, collate_fn=self.generate_batch)
            return predict_iter

        test_data, _ = self.data_process(test_file_path)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(train_file_path)  # 得到處理好的所有樣本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(val_file_path)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 構造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 開始對一個batch中的每一個樣本進行處理。
            batch_sentence.append(sen)
            batch_label.append(label)
        # print("batch_sentence:", batch_sentence)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        # print("batch_sentence1:", batch_sentence)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label


# 2022/3/3 新增
class LoadNameEntityRecognitionDataset(LoadSingleSentenceClassificationDataset):
    def __init__(self, label2id=None, word_pad_idx=0, label_pad_idx=-1, dev_split_size=None, **kwargs):
        super(LoadNameEntityRecognitionDataset, self).__init__(**kwargs)
        self.label2id = label2id
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.dev_split_size = dev_split_size

    def preprocess(self, origin_sentences, origin_labels):
        data = []
        sentences = []
        labels = []
        # print(self.vocab['[SEP]'])
        # print(self.vocab['0'])
        for line in origin_sentences:
            words = []
            word_lens = []
            for token in line:
                words.append(self.tokenizer(token))
                word_lens.append(len(token))
            # print("words:", words)
            # print("words_len:", len(words))
            # 變成單個字的列表，開頭加上[CLS]
            words = ['[CLS]'] + [item for token in words for item in token] + ['[SEP]']
            # print("words_cls:", words)
            # print("words_cls_len:", len(words))
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append(([self.vocab[word] for word in words], token_start_idxs))

        for tag in origin_labels:
            label_id = [self.label2id.get(t) for t in tag]
            labels.append(label_id)

        for sentence, label, origin_sentence in zip(sentences, labels, origin_sentences):
            data.append((sentence, label, origin_sentence))
        return data

    def load_train_val_test_data(self, train_word=None, train_label=None, test_word=None, test_label=None, predict_word=None, predict_label=None):
        if predict_word:
            predict_data = self.preprocess(predict_word, predict_label)
            pre_iter = DataLoader(predict_data, batch_size=self.batch_size,
                                  shuffle=self.is_sample_shuffle,collate_fn=self.generate_batch)
            return pre_iter
        else:
            train_data = self.preprocess(train_word, train_label)
            val_data = self.preprocess(test_word, test_label)
            train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 構造DataLoader
                                    shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
            val_iter = DataLoader(val_data, batch_size=self.batch_size,  # 構造DataLoader
                                    shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
            return train_iter, train_data, val_iter

    def generate_batch(self, data_batch):
        sentences = [x[0] for x in data_batch]
        labels = [x[1] for x in data_batch]
        origin_contents = [x[2] for x in data_batch]

        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_origin_len = max([len(s) for s in origin_contents])  # 78
        # print("max_origin_len:",max_origin_len)
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0

        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))  # 79
        batch_label_starts = []

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])  # 79
            batch_data[j][:cur_len] = sentences[j][0]  # batch_data 101, 1068, 754, 122, 121, 3299, 123
            # 找到有標籤的數據的index（[CLS]不算）
            label_start_idx = sentences[j][-1]  # len 78
            label_starts = np.zeros(max_len)  # len 79
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)  # batch_label_starts 0,1,1,1,1,1,1,,1,1,1,1
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))  # 78
        for j in range(batch_len):
            cur_tags_len = len(labels[j])  # 78
            batch_labels[j][:cur_tags_len] = labels[j]  # 00000000000000

        # original content
        batch_original_contents = np.ones((batch_len, max_origin_len), dtype=str)
        for j in range(batch_len):
            cur_original_len = len(origin_contents[j])
            batch_original_contents[j][:cur_original_len] = origin_contents[j]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return [batch_data, batch_label_starts, batch_labels, batch_original_contents]
