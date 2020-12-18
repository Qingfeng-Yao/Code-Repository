from torchnlp.datasets import imdb_dataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torchnlp.encoders.text.text_encoder import pad_tensor

import torch
from torch.utils.data import Subset

import nltk
import numpy as np

import datasets
from . import util

class IMDB_DATA:

    class Data:
        def __init__(self, data, fixed_len):
            
            texts, labels, weights, pos = [], [], [], []
            for row in datasets_iterator(data):
                texts.append(pad_tensor(row['text'], fixed_len))
                labels.append(row['label'])
                weights.append(row['weight'])
                length = row['text'].shape[0]
                pos_tensor = torch.arange(1,length+1) 
                pos.append(pad_tensor(pos_tensor, fixed_len))
            # check if weights are empty
            if weights[0].nelement() == 0:
                self.weights = torch.stack(weights).float()
            else:
                self.weights = []
                for w in weights:
                    self.weights.append(pad_tensor(w, fixed_len))
                    self.weights = torch.stack(self.weights).contiguous()
            
            self.texts = torch.stack(texts).contiguous()
            self.labels = torch.stack(labels).float()
            self.pos = torch.stack(pos).contiguous()

    def __init__(self, tokenize='spacy', normal_class=0, fixed_len=0, append_sos=True, append_eos=True, use_tfidf_weights=False):
        self.n_classes = 2 
        classes = ['pos', 'neg']
        self.normal_classes = [classes[normal_class]]
        del classes[normal_class]
        self.outlier_classes = classes

        self.trn, self.val, self.tst = load_data()

        self.trn.columns.remove('sentiment')
        self.val.columns.remove('sentiment')
        self.tst.columns.remove('sentiment')
        self.trn.columns.add('label')
        self.val.columns.add('label')
        self.tst.columns.add('label')
        self.trn.columns.add('weight')
        self.val.columns.add('weight')
        self.tst.columns.add('weight')

        train_idx_normal = []  
        train_idx_outlier = []
        for i, row in enumerate(self.trn):
            row['label'] = row.pop('sentiment')
            if row['label'] in self.normal_classes:
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                train_idx_outlier.append(i)
                row['label'] = torch.tensor(1)

            row['text'] = util.clean_text(row['text'].lower())
        # print("train_normal {}  outlier {}".format(len(train_idx_normal), len(train_idx_outlier)))

        valid_idx_normal = []
        valid_idx_outlier = []
        for i, row in enumerate(self.val):
            row['label'] = row.pop('sentiment')
            if row['label'] in self.normal_classes:
                valid_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                valid_idx_outlier.append(i)
                row['label'] = torch.tensor(1)
            row['text'] = util.clean_text(row['text'].lower())
        # print("val_normal {}  outlier {}".format(len(valid_idx_normal), len(valid_idx_outlier)))

        test_idx = []  
        for i, row in enumerate(self.tst):
            row['label'] = row.pop('sentiment')
            row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
            test_idx.append(i)
            row['text'] = util.clean_text(row['text'].lower())
        # print("test {}".format(len(test_idx)))

        self.train_set = Subset(self.trn, train_idx_normal)
        self.valid_set = Subset(self.val, valid_idx_normal)
        self.test_set = self.tst
        self.train_set_oe = Subset(self.trn, train_idx_outlier)
        self.valid_set_oe = Subset(self.val, valid_idx_outlier)

        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.valid_set, self.test_set, self.train_set_oe, self.valid_set_oe)]
        if tokenize == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=False)
        if tokenize == 'bert':
            self.encoder = util.MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir='data/bert_cache')

        # Encode
        for row in datasets_iterator(self.train_set, self.valid_set, self.test_set, self.train_set_oe, self.valid_set_oe):
            if fixed_len:
                text = self.encoder.encode(row['text'][:fixed_len])
            else:
                text = self.encoder.encode(row['text'])
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), text))
            else:
                row['text'] = text

        # Compute tf-idf weights
        if use_tfidf_weights:
            util.compute_tfidf_weights(self.train_set, self.valid_set, self.test_set, self.train_set_oe, self.valid_set_oe, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.valid_set, self.test_set, self.train_set_oe, self.valid_set_oe):
                row['weight'] = torch.empty(0)

        if fixed_len:
            self.train_set = self.Data(self.train_set, fixed_len=fixed_len)
            self.valid_set = self.Data(self.valid_set, fixed_len=fixed_len)
            self.test_set = self.Data(self.test_set, fixed_len=fixed_len)
            self.train_set_oe = self.Data(self.train_set_oe, fixed_len=fixed_len)
            self.valid_set_oe = self.Data(self.valid_set_oe, fixed_len=fixed_len)

        

def load_data():
    directory = datasets.root + 'imdb/'
    if datasets.root not in nltk.data.path:
        nltk.data.path.append(datasets.root)


    train_data, test_data = imdb_dataset(directory=directory, train=True, test=True)
           
    # print("train {}".format(len(train_data)))
    # print("test {}".format(len(test_data)))

    rng = np.random.RandomState(42)
    train_data = train_data[:]
    train_data = np.array(train_data)
    rng.shuffle(train_data)

    N_validate = int(0.1 * len(train_data))
    data_validate = train_data[-N_validate:]
    data_train = train_data[0:-N_validate]

    data_train = Dataset(data_train)
    data_validate = Dataset(data_validate)
    data_test = test_data

    return data_train, data_validate, data_test