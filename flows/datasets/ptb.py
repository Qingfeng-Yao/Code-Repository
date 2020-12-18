import numpy as np
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.encoders.text.text_encoder import pad_tensor

import datasets

class PTB_DATA:
    class Data:
        def __init__(self, data):

            self.x = data
            self.N = self.x.shape[0]


    def __init__(self):
        trn, val, tst, vocab_size = load_data()

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)
        self.vocab_size = vocab_size

        self.n_dims = self.trn.x.shape[1]


def load_data():
    directory = datasets.root + 'ptb/'

    MAX_LEN = 288
    MIN_LEN = 1

    train_data = []
    valid_data = []
    test_data = []
    for split_set in ['train', 'valid', 'test']:
        with open(directory+split_set+'.txt', encoding='utf-8') as f:
            for line in f:
                text = ' '.join(list(line.strip()))

                if split_set == 'train':
                    train_data.append(text)
                elif split_set == 'valid':
                    valid_data.append(text)
                elif split_set == 'test':
                    test_data.append(text)

    text_corpus = (train_data)
    encoder = SpacyEncoder(text_corpus)
    # print(encoder.vocab)
    # print(len(encoder.vocab))

    data_train, data_validate, data_test = [], [], []
    for i, row in enumerate(train_data):
        tokens = encoder.encode(row)
        if len(tokens) > MAX_LEN or len(tokens) < MIN_LEN:
            continue
        padded = pad_tensor(tokens, MAX_LEN)
        data_train.append(padded.cpu().data.numpy())

    for i, row in enumerate(valid_data):
        tokens = encoder.encode(row)
        if len(tokens) > MAX_LEN or len(tokens) < MIN_LEN:
            continue
        padded = pad_tensor(tokens, MAX_LEN)
        data_validate.append(padded.cpu().data.numpy())

    for i, row in enumerate(test_data):
        tokens = encoder.encode(row)
        if len(tokens) > MAX_LEN or len(tokens) < MIN_LEN:
            continue
        padded = pad_tensor(tokens, MAX_LEN)
        data_test.append(padded.cpu().data.numpy())

    data_train = np.vstack(data_train)
    data_validate = np.vstack(data_validate)
    data_test = np.vstack(data_test)

    return data_train, data_validate, data_test, len(encoder.vocab)
