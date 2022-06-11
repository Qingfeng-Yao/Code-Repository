import string
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_EOS_INDEX, DEFAULT_UNKNOWN_INDEX

def clean_text(text: str, rm_numbers=True, rm_punct=True, rm_stop_words=True, rm_short_words=True):
    """ Function to perform common NLP pre-processing tasks. """

    # make lowercase
    text = text.lower()

    # remove punctuation
    if rm_punct:
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # remove numbers
    if rm_numbers:
        text = re.sub(r'\d+', '', text)

    # remove whitespaces
    text = text.strip()

    # remove stopwords
    if rm_stop_words:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        text_list = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(text_list)

    # remove short words
    if rm_short_words:
        text_list = [w for w in text.split() if len(w) >= 3]
        text = ' '.join(text_list)

    return text

def compute_tfidf_weights(train_set, valid_set, test_set, train_set_oe, valid_set_oe, vocab_size):
    """ Compute the Tf-idf weights (based on idf vector computed from train_set, valid_set, train_set_oe and valid_set_oe)."""

    transformer = TfidfTransformer()

    # fit idf vector on train_set, valid_set, train_set_oe and valid_set_oe
    counts = np.zeros((len(train_set)+len(valid_set)+len(train_set_oe)+len(valid_set_oe), vocab_size), dtype=np.int64)
    for i, row in enumerate(train_set+valid_set+train_set_oe+valid_set_oe):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.fit_transform(counts)

    for i, row in enumerate(train_set+valid_set+train_set_oe+valid_set_oe):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())

    # compute tf-idf weights for test set (using idf vector from train_set, valid_set, train_set_oe and valid_set_oe)
    counts = np.zeros((len(test_set), vocab_size), dtype=np.int64)
    for i, row in enumerate(test_set):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.transform(counts)

    for i, row in enumerate(test_set):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())
