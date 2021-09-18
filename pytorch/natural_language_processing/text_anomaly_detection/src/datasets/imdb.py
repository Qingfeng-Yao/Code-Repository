from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets import imdb_dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights

import torch
import nltk
import numpy as np 


class IMDB_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        classes = ['pos', 'neg']

        if normal_class == -1:
            self.normal_classes = classes
            self.outlier_classes = []
        else:
            self.normal_classes = [classes[normal_class]]
            del classes[normal_class]
            self.outlier_classes = classes

        if root not in nltk.data.path:
            nltk.data.path.append(root)
        # Load the imdb dataset
        self.train_set, self.test_set = imdb_dataset(directory=root, train=True, test=True)

        # Pre-process
        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.remove('sentiment')
        self.test_set.columns.remove('sentiment')
        self.train_set.columns.add('label')
        self.test_set.columns.add('label')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')

        train_idx_normal = []  # for subsetting train_set to normal class
        for i, row in enumerate(self.train_set):
            row['label'] = row.pop('sentiment')
            if row['label'] in self.normal_classes:
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                row['label'] = torch.tensor(1)
            if clean_txt:
                row['text'] = clean_text(row['text'].lower())
            else:
                row['text'] = row['text'].lower()

        test_n_idx = [] # subsetting test_set to selected normal classes
        test_a_idx = [] # subsetting test_set to selected anomalous classes
        for i, row in enumerate(self.test_set):
            row['label'] = row.pop('sentiment')
            if row['label'] in self.normal_classes:
                test_n_idx.append(i)
            else:
                test_a_idx.append(i)
            row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
            if clean_txt:
                row['text'] = clean_text(row['text'].lower())
            else:
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
        sent_lengths_freq = sent_lengths_freq + 1
        self.length_prior = np.log(sent_lengths_freq) - np.log(sent_lengths_freq.sum())
