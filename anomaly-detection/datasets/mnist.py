import numpy as np

import datasets
from . import util
 
import torch
from torch.utils.data import Subset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class Mnist:
    def __init__(self, normal_class=0, root=datasets.root):

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                    (-0.6661464580883915, 20.108062262467364),
                    (-0.7820454743183202, 11.665100841080346),
                    (-0.7645772083211267, 12.895051191467457),
                    (-0.7253923114302238, 12.683235701611533),
                    (-0.7698501867861425, 13.103278415430502),
                    (-0.778418217980696, 10.457837397569108),
                    (-0.7129780970522351, 12.057777597673047),
                    (-0.8280402650205075, 10.581538445782988),
                    (-0.7369959242164307, 10.697039838804978)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: util.global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],
                                                                [min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
 
        train_set = MNIST(root=root, train=True, download=True, transform=transform, target_transform=target_transform)

        # Subset train_set to normal class
        train_idx_normal = util.get_target_label_idx(train_set.targets.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MNIST(root=root, train=False, download=True, transform=transform, target_transform=target_transform)