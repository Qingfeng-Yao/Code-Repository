from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN

import torch
from torch.utils.data import Subset

import nltk
from sklearn.datasets import fetch_20newsgroups
import numpy as np

import datasets
from . import util

class NEWSGROUP_DATA:

    def __init__(self, normal_class=0, min_count=3, max_seq=450, tokenize='spacy', append_sos=False, append_eos=False, use_tfidf_weights=False):
        self.n_classes = 2 
        classes = list(range(6))
        groups = [
            ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
             'comp.windows.x'],
            ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
            ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
            ['misc.forsale'],
            ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
            ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']
        ]
        self.normal_classes = groups[normal_class]
        self.outlier_classes = []
        del classes[normal_class]
        for i in classes:
            self.outlier_classes += groups[i]

        self.trn, self.val, self.tst = load_data()

        self.trn.columns.add('weight')
        self.val.columns.add('weight')
        self.tst.columns.add('weight')

        train_idx_normal = []  
        train_idx_outlier = []
        for i, row in enumerate(self.trn):
            if row['label'] in self.normal_classes:
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                train_idx_outlier.append(i)
                row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()
        print("train_normal {}  outlier {}".format(len(train_idx_normal), len(train_idx_outlier)))

        valid_idx_normal = []
        valid_idx_outlier = []
        for i, row in enumerate(self.val):
            if row['label'] in self.normal_classes:
                valid_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                valid_idx_outlier.append(i)
                row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()
        print("val_normal {}  outlier {}".format(len(valid_idx_normal), len(valid_idx_outlier)))

        test_idx = []  
        test_normal, test_anomaly = 0, 0
        for i, row in enumerate(self.tst):
            if row['label'] in self.normal_classes:
                row['label'] = torch.tensor(0)
                test_normal += 1
            else:
                row['label'] = torch.tensor(1)
                test_anomaly += 1
            test_idx.append(i)
            row['text'] = row['text'].lower()
        print("test {}, test_normal {}, test_anomaly {}".format(len(test_idx), test_normal, test_anomaly))

        self.train_set = Subset(self.trn, train_idx_normal)
        self.valid_set = Subset(self.val, valid_idx_normal)
        self.test_set = self.tst

        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.valid_set, self.test_set)]
        if tokenize == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=min_count, append_eos=append_eos)

        # Encode
        for row in datasets_iterator(self.train_set, self.valid_set, self.test_set):
            text = self.encoder.encode(row['text'][:max_seq])
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), text))
            else:
                row['text'] = text

        # Compute tf-idf weights
        if use_tfidf_weights:
            util.compute_tfidf_weights(self.train_set, self.valid_set, self.test_set, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.valid_set, self.test_set):
                row['weight'] = torch.empty(0)


def load_data():
    directory = datasets.root + 'newsgroup/'
    if datasets.root not in nltk.data.path:
        nltk.data.path.append(datasets.root)

    train_data = []
    test_data = []
    for split_set in ['train', 'test']:
        dataset = fetch_20newsgroups(data_home=directory, subset=split_set, remove=('headers', 'footers', 'quotes'))

        for id in range(len(dataset.data)):
            text = util.clean_text(dataset.data[id])
            if len(text) == 0:
                continue
            labels = dataset.target_names[int(dataset.target[id])]
            if split_set == 'train':
                train_data.append({
                    'text': text,
                    'label': labels,
                })
            elif split_set == 'test':
                test_data.append({
                    'text': text,
                    'label': labels,
                })
    # print("train {}".format(len(train_data)))
    # print("test {}".format(len(test_data)))

    rng = np.random.RandomState(42)
    train_data = np.array(train_data)
    rng.shuffle(train_data)

    N_validate = int(0.1 * len(train_data))
    data_validate = train_data[-N_validate:]
    data_train = train_data[0:-N_validate]

    data_train = Dataset(data_train)
    data_validate = Dataset(data_validate)
    data_test = Dataset(test_data)

    return data_train, data_validate, data_test