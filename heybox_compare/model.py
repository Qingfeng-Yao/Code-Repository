import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
 
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

class SelfAttention(nn.Module):

    def __init__(self, hidden_size, attention_size=100, n_attention_heads=1):
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

    def __init__(self, pretrained_model, attention_size=100, n_attention_heads=1):
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

        return cosine_dists, context_weights, A

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
        text, _ = data
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
    return centers # (n_attention_heads, embedding_size)