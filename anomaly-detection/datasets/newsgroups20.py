import torch
from torch.utils.data import Subset
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text import SpacyEncoder

from sklearn.datasets import fetch_20newsgroups
import nltk

import datasets
from . import util
 
class Newsgroups20:
    def __init__(self, normal_class=0, root=datasets.root):

        self.n_classes = 2  # 0: normal, 1: outlier
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

        self.train_set, self.test_set = newsgroups20_dataset(directory=root, train=True, test=True)

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

def newsgroups20_dataset(directory='../data', train=False, test=False):
    """
    Load the 20 Newsgroups dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:

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

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
