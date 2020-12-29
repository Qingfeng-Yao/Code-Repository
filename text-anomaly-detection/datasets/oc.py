import torch
from torch.utils.data import Subset
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text import SpacyEncoder

from sklearn.datasets import fetch_20newsgroups

import nltk
from nltk.corpus import reuters

import datasets
from . import util

class OC_DATA:

    def __init__(self, dataset, normal_class=0):

        self.n_classes = 2  # 0: normal, 1: outlier

        if dataset == "reuters":
            classes = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']
            self.normal_classes = [classes[normal_class]]
            del classes[normal_class]
            self.outlier_classes = classes

            self.train_set, self.test_set = reuters_dataset(dataset_str=dataset)
            train_idx_normal = []  # for subsetting train_set to normal class
            for i, row in enumerate(self.train_set):
                if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
                    train_idx_normal.append(i)
                    row['label'] = torch.tensor(0)
                else:
                    row['label'] = torch.tensor(1)
                row['text'] = row['text'].lower()

            test_idx = []  # for subsetting test_set to selected normal and anomalous classes
            for i, row in enumerate(self.test_set):
                if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
                    test_idx.append(i)
                    row['label'] = torch.tensor(0)
                elif any(label in self.outlier_classes for label in row['label']) and (len(row['label']) == 1):
                    test_idx.append(i)
                    row['label'] = torch.tensor(1)
                else:
                    row['label'] = torch.tensor(1)
                row['text'] = row['text'].lower()

            # Subset train_set to normal class
            self.train_set = Subset(self.train_set, train_idx_normal)
            # Subset test_set to selected normal and anomalous classes
            self.test_set = Subset(self.test_set, test_idx)


        if dataset == "newsgroup":
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

            self.train_set, self.test_set = newsgroups20_dataset(dataset_str=dataset)
            train_idx_normal = []  # for subsetting train_set to normal class
            for i, row in enumerate(self.train_set):
                if row['label'] in self.normal_classes:
                    train_idx_normal.append(i)
                    row['label'] = torch.tensor(0)
                else:
                    row['label'] = torch.tensor(1)
                row['text'] = row['text'].lower()

            for i, row in enumerate(self.test_set):
                row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
                row['text'] = row['text'].lower()

            # Subset train_set to normal class
            self.train_set = Subset(self.train_set, train_idx_normal)


        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=False)
        for row in datasets_iterator(self.train_set, self.test_set):
            row['text'] = self.encoder.encode(row['text'])




def reuters_dataset(dataset_str):
    if datasets.root not in nltk.data.path:
        nltk.data.path.append(datasets.root)
    directory = datasets.root + dataset_str + '/'
    # nltk.download('reuters', download_dir=directory)
    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    doc_ids = reuters.fileids()

    ret = []

    for split_set in ['train', 'test']:
        split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))
        examples = []

        for id in split_set_doc_ids:
            text = util.clean_text(reuters.raw(id))
            labels = reuters.categories(id)

            examples.append({
                'text': text,
                'label': labels,
            })

        ret.append(Dataset(examples))

    return tuple(ret)

def newsgroups20_dataset(dataset_str):
    if datasets.root not in nltk.data.path:
        nltk.data.path.append(datasets.root)
    directory = datasets.root + dataset_str + '/'

    ret = []

    for split_set in ['train', 'test']:
        dataset = fetch_20newsgroups(data_home=directory, subset=split_set, remove=('headers', 'footers', 'quotes'))
        examples = []

        for id in range(len(dataset.data)):
            text = util.clean_text(dataset.data[id])
            label = dataset.target_names[int(dataset.target[id])]

            if text:
                examples.append({
                    'text': text,
                    'label': label
                })

        ret.append(Dataset(examples))

    return tuple(ret)