import torch
from torch.utils.data import Subset
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text import SpacyEncoder

import datasets
from . import util

import nltk
from nltk.corpus import reuters

class Reuters:
    def __init__(self, normal_class=6, root=datasets.root):

        self.n_classes = 2  # 0: normal, 1: outlier
        classes = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']
        self.normal_classes = [classes[normal_class]]
        del classes[normal_class]
        self.outlier_classes = classes

        self.train_set, self.test_set = reuters_dataset(directory=root, train=True, test=True)

        train_idx_normal = []
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

        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=False)

        for row in datasets_iterator(self.train_set, self.test_set):
            row['text'] = self.encoder.encode(row['text'])



def reuters_dataset(directory='../data', train=True, test=False):
    """
    Load the Reuters-21578 dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    nltk.download('reuters', download_dir=directory)
    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    doc_ids = reuters.fileids()

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:

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

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
