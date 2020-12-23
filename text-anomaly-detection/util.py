import torch
import os
import numpy as np
import sklearn.metrics as sk

def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.to(device)
    return data

def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class OODCorpus(object):
    def __init__(self, path, dictionary, char=False):
        """
        :param path: path to train, val, or test data
        :param dictionary: existing dictionary of words constructed with Corpus class on in-dist
        :param char: if True, return character-level data
        """
        self.dictionary = dictionary
        self.data_words, self.data = self.tokenize(path, char)

    def tokenize(self, path, char=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        corpus = []
        ids = []
        with open(path, 'r') as f:
            for line in f:
                if len(line) == 1:  # end of example
                    if char:
                        corpus.append('<eos>')
                        ids.append(self.dictionary.word2idx['<eos>'])
                    else:
                        corpus.append('<eos>')
                        ids.append(self.dictionary.word2idx['<eos>'])
                    continue
                word = line.split('\t')[1]
                if char:
                    if word not in self.dictionary.word2idx.keys():
                        word = '<unk>'
                    corpus.extend(list(word))
                    corpus.append('_')
                    ids.extend([self.dictionary.word2idx[char] for char in word])
                    ids.append(self.dictionary.word2idx['_'])
                else:
                    corpus.append(word)
                    ids.append(self.dictionary.word2idx.get(word, self.dictionary.word2idx['<unk>']))

        return corpus, torch.LongTensor(ids)

def show_performance(pos, neg, recall_level=0.95):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc = get_measures(pos[:], neg[:], recall_level)
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)

    return auroc
