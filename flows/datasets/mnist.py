import gzip
import pickle
import numpy as np

import datasets

from . import util


class MNIST_DATA:

    alpha = 1.0e-6

    class Data:

        def __init__(self, data, logit, dequantize, rng):

            x = self._dequantize(
                data[0], rng) if dequantize else data[0]  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x  # logit
            self.labels = data[1]  # numeric labels
            self.y = util.one_hot_encode(self.labels,
                                         10)  # 1-hot encoded labels
            self.N = self.x.shape[0]  # number of datapoints
            self.x = self.x.astype('float32')
            self.y = self.y.astype('int')

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        @staticmethod
        def _logit_transform(x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            return util.logit(MNIST_DATA.alpha + (1 - 2 * MNIST_DATA.alpha) * x)

    def __init__(self, logit=True, dequantize=True):

        # load dataset
        f = gzip.open(datasets.root + 'mnist/mnist.pkl.gz', 'rb')
        trn, val, tst = pickle.load(f, encoding='latin1')
        f.close()

        rng = np.random.RandomState(42)
        self.trn = self.Data(trn, logit, dequantize, rng)
        self.val = self.Data(val, logit, dequantize, rng)
        self.tst = self.Data(tst, logit, dequantize, rng)

        self.n_dims = self.trn.x.shape[1]
        self.n_labels = self.trn.y.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims))] * 2