from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights

import torch
import nltk
import numpy as np 
import os


class Newsgroups20_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False, max_seq_len_prior=None):
        super().__init__(root)

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
        short_group_names = ['comp', 'rec', 'sci', 'misc', 'pol', 'rel']
        self.subset = short_group_names[normal_class]

        self.normal_classes = groups[normal_class]
        self.outlier_classes = []
        del classes[normal_class]
        for i in classes:
            self.outlier_classes += groups[i]

        # Load the 20 Newsgroups dataset
        self.train_set, self.test_set = newsgroups20_dataset(directory=root, train=True, test=True, clean_txt=clean_txt, groups=groups, short_group_names=short_group_names)

        # Pre-process
        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')

        train_idx_normal = []  # for subsetting train_set to normal class
        for i, row in enumerate(self.train_set):
            if row['label'] in self.normal_classes:
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()

        test_n_idx = [] # subsetting test_set to selected normal classes
        test_a_idx = [] # subsetting test_set to selected anomalous classes
        for i, row in enumerate(self.test_set):
            if row['label'] in self.normal_classes:
                test_n_idx.append(i)
            else:
                test_a_idx.append(i)
            row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
            row['text'] = row['text'].lower()

        # Subset train_set to normal class
        self.train_set = Subset(self.train_set, train_idx_normal)
        # Subset test_set to selected normal classes
        self.test_n_set = Subset(self.test_set, test_n_idx)
        # Subset test_set to selected anomalous classes
        self.test_a_set = Subset(self.test_set, test_a_idx)

        # Make corpus and set encoder
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        if tokenizer == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        if tokenizer == 'bert':
            self.encoder = MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir=root)

        # Encode
        self.max_seq_len = 0
        for row in datasets_iterator(self.train_set, self.test_set):
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
            else:
                row['text'] = self.encoder.encode(row['text'])
            if len(row['text'])>self.max_seq_len:
                self.max_seq_len = len(row['text'])

        # Compute tf-idf weights
        if use_tfidf_weights:
            compute_tfidf_weights(self.train_set, self.test_set, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.test_set):
                row['weight'] = torch.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['index'] = i
        for i, row in enumerate(self.test_set):
            row['index'] = i

        # length prior
        sent_lengths = [len(row['text']) for row in self.train_set]
        sent_lengths_freq = np.bincount(np.array(sent_lengths))
        sent_lengths_freq = np.concatenate((sent_lengths_freq, np.array((max_seq_len_prior-max(sent_lengths))*[0])), axis=0)
        sent_lengths_freq = sent_lengths_freq + 1
        self.length_prior = np.log(sent_lengths_freq) - np.log(sent_lengths_freq.sum())


def newsgroups20_dataset(directory='../data', train=False, test=False, clean_txt=False, groups=None, short_group_names=None):
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
    
    file_export = True
    export_path = os.path.join(directory, 'newsgroups20')
    if os.path.exists(export_path):
        file_export = False

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:

        dataset = fetch_20newsgroups(data_home=directory, subset=split_set, remove=('headers', 'footers', 'quotes'))
        examples = []
        corpus = {}
        if file_export:
            for g in short_group_names:
                corpus[g] = ''
        
        for id in range(len(dataset.data)):
            label = dataset.target_names[int(dataset.target[id])]
            if clean_txt:
                text = clean_text(dataset.data[id])
            else:
                text = ' '.join(word_tokenize(dataset.data[id]))
            
            if text:
                examples.append({
                    'text': text,
                    'label': label
                })
                if file_export:
                    for i, g in enumerate(groups):
                        if label in g:
                            corpus[short_group_names[i]] += f'\n\n{text}'
                

        ret.append(Dataset(examples))

        if file_export:
            full_path = os.path.join(export_path, split_set)
            for g in short_group_names:

                if not os.path.exists(full_path):
                    os.makedirs(full_path)

                with open(os.path.join(full_path,f'{g}.txt'), 'w') as f:
                    f.write(corpus[g])

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
