import string
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_EOS_INDEX, DEFAULT_UNKNOWN_INDEX
from pytorch_pretrained_bert import BertTokenizer

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

class MyBertTokenizer(BertTokenizer):
    """ Patch of pytorch_pretrained_bert.BertTokenizer to fit torchnlp TextEncoder() interface. """

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, append_eos=False):
        super().__init__(vocab_file, do_lower_case=do_lower_case)
        self.append_eos = append_eos

        self.itos = list(self.vocab.keys())
        self.stoi = {token: index for index, token in enumerate(self.itos)}

        self.vocab = self.itos
        self.vocab_size = len(self.vocab)

    def encode(self, text, eos_index=DEFAULT_EOS_INDEX, unknown_index=DEFAULT_UNKNOWN_INDEX):
        """ Returns a :class:`torch.LongTensor` encoding of the `text`. """
        text = self.tokenize(text)
        unknown_index = self.stoi['[UNK]']  # overwrite unknown_index according to BertTokenizer vocab
        vector = [self.stoi.get(token, unknown_index) for token in text]
        if self.append_eos:
            vector.append(eos_index)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        """ Given a :class:`torch.Tensor`, returns a :class:`str` representing the decoded text.
        Note that, depending on the tokenization method, the decoded version is not guaranteed to be
        the original text.
        """
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)

def compute_tfidf_weights(train_set, valid_set, test_set, vocab_size):
    """ Compute the Tf-idf weights (based on idf vector computed from train_set and valid_set)."""

    transformer = TfidfTransformer()

    # fit idf vector on train set and valid set
    counts = np.zeros((len(train_set)+len(valid_set), vocab_size), dtype=np.int64)
    for i, row in enumerate(train_set+valid_set):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.fit_transform(counts)

    for i, row in enumerate(train_set+valid_set):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())

    # compute tf-idf weights for test set (using idf vector from train set and valid set)
    counts = np.zeros((len(test_set), vocab_size), dtype=np.int64)
    for i, row in enumerate(test_set):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.transform(counts)

    for i, row in enumerate(test_set):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())
