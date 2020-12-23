import numpy as np
from collections import Counter

import torch

import datasets

class LM_DATA:
    class Data:
        def __init__(self, data):

            self.x = data
            self.N = self.x.shape[0]


    def __init__(self, dataset, dictionary=None):
        self.dictionary = Dictionary() if dictionary is None else dictionary
        self.new_dict = True if dictionary is None else False
        trn, val, tst  = self.load_data(dataset)
        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)
        self.vocab_size = len(self.dictionary)


    def load_data(self, dataset):
        directory = datasets.root + dataset + '/'

        for split_set in ['train', 'valid', 'test']:
            with open(directory+split_set+'.txt', encoding='utf-8') as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    if self.new_dict:
                        for word in words:
                            self.dictionary.add_word(word)

            with open(directory+split_set+'.txt', encoding='utf-8') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx.get(word, self.dictionary.word2idx['<unk>'])
                        token += 1

            if split_set == 'train':
                train_data = ids
            elif split_set == 'valid':
                valid_data = ids
            elif split_set == 'test':
                test_data = ids

        return train_data, valid_data, test_data

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
