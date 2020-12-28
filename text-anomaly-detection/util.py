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

class CorpusWikiTextChar(object):
    def __init__(self, path, dictionary):
        """
        :param path: path to train, val, or test data
        :param dictionary: If None, create new dictionary. Else, use the given dictionary.
        """
        self.dictionary = dictionary
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        ids = []
        with open(path, 'r') as f:
            for line in f:
                if len(line) == 2:  # end of example
                    continue

                words = line.split()
                if words[0] != '=':
                    for i in range(len(words)):
                        word = words[i]
                        for char in [char for char in word]:
                            if char in self.dictionary.word2idx.keys():
                                ids.append(self.dictionary.word2idx[char])
                            else:
                                ids.append(self.dictionary.word2idx['<unk>'])

                        if i < len(words) - 1:
                            ids.append(self.dictionary.word2idx['_'])
                    ids.append(self.dictionary.word2idx['<eos>'])

        return torch.LongTensor(ids)

class OODCorpus(object):
    def __init__(self, path, dictionary, char_level=False):
        """
        :param path: path to data
        :param dictionary: existing dictionary of words constructed with Corpus class on in-dist
        :param char_level: if True, return character-level data
        """
        self.dictionary = dictionary
        self.data = self.tokenize(path, char_level)

    def tokenize(self, path, char_level=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        ids = []
        with open(path, 'r') as f:
            for line in f:
                if len(line) == 1: # end of example
                    ids.append(self.dictionary.word2idx['<eos>'])
                    continue

                if len(line.split('\t')) > 1:
                    word = line.split('\t')[1]
                    if char_level:
                        for char in [char for char in word]:
                            if char in self.dictionary.word2idx.keys():
                                ids.append(self.dictionary.word2idx[char])
                            else:
                                ids.append(self.dictionary.word2idx['<unk>'])
                        ids.append(self.dictionary.word2idx['_'])

                    else:
                        ids.append(self.dictionary.word2idx.get(word, self.dictionary.word2idx['<unk>']))

        return torch.LongTensor(ids)

def show_performance(pos, neg, recall_level=0.95):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    # auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)
    auroc, aupr = get_measures(pos[:], neg[:], recall_level)
    # print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    # fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    # return auroc, aupr, fpr
    return auroc, aupr

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                 'its last element does not correspond to sum')
    return out