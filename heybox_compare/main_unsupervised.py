import argparse
import numpy as np
import copy 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from pytorch_pretrained_bert.modeling import BertModel

import datasets

parser = argparse.ArgumentParser(description='spam detection using unsupervised learning(label representation+cos)')
parser.add_argument(
    '--dataset',
    default='Heyspam',
    help='Heyspam')
parser.add_argument(
    '--normal-class', type=int, default=-1, help='specify the normal class of the dataset(all other classes are considered anomalous). if -1, then train all classes')
args = parser.parse_args()

dataset = getattr(datasets, args.dataset)(args.normal_class, is_deep=True, is_jieba=False)
idx_normal, idx_spam = [], []  
for i, row in enumerate(dataset.train_set):
    if row['label'] == torch.tensor(0):
        idx_normal.append(i)
    elif row['label'] == torch.tensor(1):
        idx_spam.append(i)

normal_set = Subset(dataset.train_set, idx_normal)
spam_set = Subset(dataset.train_set, idx_spam)

def collate_fn(batch):
    """ list of tensors to a batch tensors """
    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    text_batch, _ = stack_and_pad_tensors([row['text'] for row in batch])
    label_batch = torch.stack([row['label'] for row in batch])

    return transpose(text_batch), label_batch.float()

normal_sampler = BucketBatchSampler(normal_set, batch_size=10, drop_last=False,
                                           sort_key=lambda r: len(r['text']))
spam_sampler = BucketBatchSampler(spam_set, batch_size=10, drop_last=False,
                                    sort_key=lambda r: len(r['text']))
test_sampler = BucketBatchSampler(dataset.test_set, batch_size=1, drop_last=False,
                                    sort_key=lambda r: len(r['text']))

normal_loader = torch.utils.data.DataLoader(dataset=normal_set, batch_sampler=normal_sampler, collate_fn=collate_fn)
spam_loader = torch.utils.data.DataLoader(dataset=spam_set, batch_sampler=spam_sampler, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(dataset=dataset.test_set, batch_sampler=test_sampler, collate_fn=collate_fn)

class BERT(nn.Module):
    """Class for loading pretrained BERT model."""

    def __init__(self, pretrained_model_name='bert-base-chinese', cache_dir='data/bert_cache', embedding_reduction='none'):
        super().__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir)
        self.embedding = self.bert.embeddings
        # print(self.embedding.word_embeddings.weight.data.shape) # torch.Size([21128, 768]) 
        self.embedding_size = self.embedding.word_embeddings.embedding_dim
        # print("embedding_size {}".format(self.embedding_size)) # 768

        self.reduction = embedding_reduction

        # Remove BERT model parameters from optimization
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x.shape = (sentence_length, batch_size)

        self.bert.eval()  # make sure bert is in eval() mode
        hidden, _ = self.bert(x.transpose(0, 1), output_all_encoded_layers=False)  # output only last layer
        # hidden.shape = (batch_size, sentence_length, hidden_size)

        # Change to hidden.shape = (sentence_length, batch_size, hidden_size) align output with word embeddings
        embedded = hidden.transpose(0, 1)

        if self.reduction == 'mean':
            embedded = torch.mean(embedded, dim=0)
            embedded = embedded / torch.norm(embedded, p=2, dim=1, keepdim=True).clamp(min=1e-08)
            embedded[torch.isnan(embedded)] = 0

        return embedded

Model = BERT()

x_normal, x_spam, x_test = (), (), ()
for data in normal_loader:
    text, _ = data
    # text.shape = (sentence_length, batch_size)

    x_batch = Model(text)
    # X_batch.shape = (sentence_length, batch_size, embedding_size)

    # compute mean and normalize
    x_batch = torch.mean(x_batch, dim=0)
    x_batch = x_batch / torch.norm(x_batch, p=2, dim=1, keepdim=True).clamp(min=1e-08)
    x_batch[torch.isnan(x_batch)] = 0
    # x_batch.shape = (batch_size, embedding_size)

    x_normal += (x_batch.data.numpy(),)

x_normal = np.concatenate(x_normal)
print(x_normal.shape)
kmeans = KMeans(n_clusters=1).fit(x_normal)
normal_centers = kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, ord=2, axis=1, keepdims=True)
print(normal_centers.shape) # (1,768)
for data in spam_loader:
    text, _ = data
    # text.shape = (sentence_length, batch_size)

    x_batch = Model(text)
    # X_batch.shape = (sentence_length, batch_size, embedding_size)

    # compute mean and normalize
    x_batch = torch.mean(x_batch, dim=0)
    x_batch = x_batch / torch.norm(x_batch, p=2, dim=1, keepdim=True).clamp(min=1e-08)
    x_batch[torch.isnan(x_batch)] = 0
    # x_batch.shape = (batch_size, embedding_size)

    x_spam += (x_batch.data.numpy(),)

x_spam = np.concatenate(x_spam)
print(x_spam.shape)
kmeans = KMeans(n_clusters=1).fit(x_spam)
spam_centers = kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, ord=2, axis=1, keepdims=True)
print(spam_centers.shape) # (1,768)
for data in test_loader:
    text, _ = data
    # text.shape = (sentence_length, batch_size)

    x_batch = Model(text)
    # X_batch.shape = (sentence_length, batch_size, embedding_size)

    # compute mean and normalize
    x_batch = torch.mean(x_batch, dim=0)
    x_batch = x_batch / torch.norm(x_batch, p=2, dim=1, keepdim=True).clamp(min=1e-08)
    x_batch[torch.isnan(x_batch)] = 0
    # x_batch.shape = (batch_size, embedding_size)

    x_test += (x_batch.data.numpy(),)

x_test = np.concatenate(x_test)
print(x_test.shape)

def calcSimilarity(r_x, a_x):
    return r_x @ a_x.T

similarity = calcSimilarity(x_test, np.concatenate((normal_centers, spam_centers),axis=0))

for i, p in enumerate(similarity):
    print(p)
    print(dataset.text_test[i])
    print("\n")



        