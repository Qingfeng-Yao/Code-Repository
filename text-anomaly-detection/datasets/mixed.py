import torch

from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN

import nltk
from nltk.corpus import reuters
import numpy as np
from sklearn.datasets import fetch_20newsgroups

import datasets
from . import util

class MIXED_DATA:

    def __init__(self, tokenize='spacy', normal='reuters', outlier='newsgroup', append_sos=True, append_eos=True):
        
        self.normal = normal
        self.outlier = outlier

        self.trn, self.val, self.tst_n = load_data(name=normal, is_normal=True)
        self.tst_o = load_data(name=outlier, is_normal=False)

        self.trn.columns.add('weight')
        self.val.columns.add('weight')
        self.tst_n.columns.add('weight')
        self.tst_o.columns.add('weight')

        for i, row in enumerate(self.trn):
            row['label'] = torch.tensor(0)
            row['text'] = row['text'].lower()

        for i, row in enumerate(self.val):
            row['label'] = torch.tensor(0)
            row['text'] = row['text'].lower()

        for i, row in enumerate(self.tst_n):
            row['label'] = torch.tensor(0)
            row['text'] = row['text'].lower()

        for i, row in enumerate(self.tst_o):
            row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()

        self.tst = Dataset(np.hstack((self.tst_n.rows, self.tst_o.rows)))
        
        print("train {} val {} test {}".format(len(self.trn), len(self.val), len(self.tst)))

        text_corpus = [row['text'] for row in datasets_iterator(self.trn, self.val, self.tst)]
        if tokenize == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        if tokenize == 'bert':
            self.encoder = util.MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir='data/bert_cache')

        # Encode
        for row in datasets_iterator(self.trn, self.val, self.tst):
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
            else:
                row['text'] = self.encoder.encode(row['text'])

        for row in datasets_iterator(self.trn, self.val, self.tst):
                row['weight'] = torch.empty(0)

        self.train_set, self.valid_set, self.test_set = self.trn, self.val, self.tst

def load_data(name='reuters', is_normal=True):

    if name == 'reuters':
        directory = datasets.root + 'reuters/'
        # nltk.download('reuters', download_dir=directory)
        if directory not in nltk.data.path:
            nltk.data.path.append(directory)
        doc_ids = reuters.fileids()
        data = []
        for split_set in ['train', 'test']:
            split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))

            for id in split_set_doc_ids:
                text = util.clean_text(reuters.raw(id))
                if len(text) == 0:
                    continue
                labels = reuters.categories(id)
                data.append({
                    'text': text,
                    'label': labels,
                })

    elif name == 'newsgroup':
        directory = datasets.root + 'newsgroup/'
        if datasets.root not in nltk.data.path:
            nltk.data.path.append(datasets.root)

        data = []
        for split_set in ['train', 'test']:
            dataset = fetch_20newsgroups(data_home=directory, subset=split_set, remove=('headers', 'footers', 'quotes'))

            for id in range(len(dataset.data)):
                text = util.clean_text(dataset.data[id])
                if len(text) == 0:
                    continue
                labels = dataset.target_names[int(dataset.target[id])]
                data.append({
                    'text': text,
                    'label': labels,
                })

    if is_normal:
        rng = np.random.RandomState(42)
        data = np.array(data)
        rng.shuffle(data)

        N_test = int(0.1 * len(data))
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1 * len(data_train))
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        data_train = Dataset(data_train)
        data_validate = Dataset(data_validate)
        data_test = Dataset(data_test)

        return data_train, data_validate, data_test
    else:
        return Dataset(data)
        
                