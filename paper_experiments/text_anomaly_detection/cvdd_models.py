import numpy as np
import math
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel

## ---------------
## 嵌入相关
class MyEmbedding(nn.Embedding):
    """Embedding base class."""

    def __init__(self, vocab_size, embedding_size, update_embedding=False, reduction='mean', use_tfidf_weights=False, normalize=True):
        super().__init__(vocab_size, embedding_size)

        # Check if choice of reduction is valid
        assert reduction in ('none', 'mean', 'max', 'sum')

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight.requires_grad = update_embedding
        self.reduction = reduction
        self.use_tfidf_weights = use_tfidf_weights
        self.normalize = normalize

    def forward(self, x, weights=None):
        # x.shape = (sentence_length, batch_size)
        # weights.shape = (sentence_length, batch_size)

        embedded = super().forward(x)
        # embedded.shape = (sentence_length, batch_size, embedding_size)

        # Reduce representation if specified to (weighted) mean of document word vector embeddings over sentence_length
        #   'mean' : (weighted) mean of document word vector embeddings over sentence_length
        #   'max'  : max-pooling of document word vector embedding dimensions over sentence_length
        # After reduction: embedded.shape = (batch_size, embedding_size)
        if self.reduction != 'none':
            if self.reduction == 'mean':
                if self.use_tfidf_weights:
                    # compute tf-idf weighted mean if specified
                    embedded = torch.sum(embedded * weights.unsqueeze(2), dim=0)
                else:
                    embedded = torch.mean(embedded, dim=0)

            if self.reduction == 'max':
                embedded, _ = torch.max(embedded, dim=0)

            if self.reduction == 'sum':
                embedded = torch.sum(embedded, dim=0)

            if self.normalize:
                embedded = embedded / torch.norm(embedded, p=2, dim=1, keepdim=True).clamp(min=1e-08)
                embedded[torch.isnan(embedded)] = 0

        return embedded

class BERT(nn.Module):
    """Class for loading pretrained BERT model."""

    def __init__(self, update_embedding=False, reduction='mean', use_tfidf_weights=False, normalize=True, pretrained_model_name='bert-base-uncased', cache_dir='data/bert_cache'):
        super().__init__()

        # Check if choice of pretrained model is valid
        assert pretrained_model_name in ('bert-base-uncased', 'bert-large-uncased', 'bert-base-cased')

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir)
        self.embedding = self.bert.embeddings
        self.embedding_size = self.embedding.word_embeddings.embedding_dim
        self.reduction = reduction
        self.update_embedding = update_embedding
        self.use_tfidf_weights = use_tfidf_weights
        self.normalize = normalize

        if not self.update_embedding:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, x, weights=None):
        # x.shape = (sentence_length, batch_size)
        # weights.shape = (sentence_length, batch_size)

        if not self.update_embedding:
            self.bert.eval()  
        hidden, _ = self.bert(x.transpose(0, 1), output_all_encoded_layers=False)  # output only last layer
        # hidden.shape = (batch_size, sentence_length, hidden_size)

        # Change to hidden.shape = (sentence_length, batch_size, hidden_size) align output with word embeddings
        embedded = hidden.transpose(0, 1)

        if self.reduction != 'none':
            if self.reduction == 'mean':
                if self.use_tfidf_weights:
                    # compute tf-idf weighted mean if specified
                    embedded = torch.sum(embedded * weights.unsqueeze(2), dim=0)
                else:
                    embedded = torch.mean(embedded, dim=0)

            if self.reduction == 'max':
                embedded, _ = torch.max(embedded, dim=0)

            if self.reduction == 'sum':
                embedded = torch.sum(embedded, dim=0)

            if self.normalize:
                embedded = embedded / torch.norm(embedded, p=2, dim=1, keepdim=True).clamp(min=1e-08)
                embedded[torch.isnan(embedded)] = 0

        return embedded

def initialize_context_vectors(net, train_loader, device):
    """
    Initialize the context vectors from an initial run of k-means++ on simple average sentence embeddings

    Returns
    -------
    centers : ndarray, [n_clusters, n_features]
    """

    print('Initialize context vectors...')

    # Get vector representations
    X = ()
    for data in train_loader:
        text, _, _ = data
        text = text.to(device)
        # text.shape = (sentence_length, batch_size)

        X_batch = net.pretrained_model(text)
        # X_batch.shape = (sentence_length, batch_size, embedding_size)

        # compute mean and normalize
        X_batch = torch.mean(X_batch, dim=0)
        X_batch = X_batch / torch.norm(X_batch, p=2, dim=1, keepdim=True).clamp(min=1e-08)
        X_batch[torch.isnan(X_batch)] = 0
        # X_batch.shape = (batch_size, embedding_size)

        X += (X_batch.cpu().data.numpy(),)

    X = np.concatenate(X)
    n_attention_heads = net.n_attention_heads

    kmeans = KMeans(n_clusters=n_attention_heads).fit(X)
    centers = kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, ord=2, axis=1, keepdims=True)

    print('Context vectors initialized.')

    return centers

## -----------------

## -----------------
## 文本表示相关
class SelfAttention(nn.Module):

    def __init__(self, hidden_size, attention_size=150, n_attention_heads=3):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads

        self.W1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_attention_heads, bias=False)

    def forward(self, hidden):
        # hidden.shape = (sentence_length, batch_size, hidden_size)

        # Change to hidden.shape = (batch_size, sentence_length, hidden_size)
        hidden = hidden.transpose(0, 1)

        x = torch.tanh(self.W1(hidden))
        # x.shape = (batch_size, sentence_length, attention_size)

        x = F.softmax(self.W2(x), dim=1)  # softmax over sentence_length
        # x.shape = (batch_size, sentence_length, n_attention_heads)

        A = x.transpose(1, 2)
        M = A @ hidden
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)

        return M, A

class CVDDNet(nn.Module):
    def __init__(self, pretrained_model, attention_size=150, n_attention_heads=3):
        super().__init__()

        # Load pretrained model (which provides a hidden representation per word, e.g. word vector or language model)
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size

        # Set self-attention module
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.self_attention = SelfAttention(hidden_size=self.hidden_size,
                                            attention_size=attention_size,
                                            n_attention_heads=n_attention_heads)

        # Model parameters
        self.c = nn.Parameter((torch.rand(1, n_attention_heads, self.hidden_size) - 0.5) * 2)
        self.cosine_sim = nn.CosineSimilarity(dim=2)

        # Temperature parameter alpha
        self.alpha = 0.0

    def forward(self, x):
        # x.shape = (sentence_length, batch_size)

        hidden = self.pretrained_model(x)
        # hidden.shape = (sentence_length, batch_size, hidden_size)

        M, A = self.self_attention(hidden)
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)

        cosine_dists = 0.5 * (1 - self.cosine_sim(M, self.c))
        context_weights = F.softmax(-self.alpha * cosine_dists, dim=1)

        return cosine_dists, context_weights