import torch
import torch.nn as nn
import torch.nn.functional as F


class MyEmbedding(nn.Embedding):
    """Embedding base class."""

    def __init__(self, vocab_size, embedding_size, update_embedding=False, reduction='none', normalize=False):
        super().__init__(vocab_size, embedding_size)

        # Check if choice of reduction is valid
        assert reduction in ('none', 'mean', 'max')

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight.requires_grad = update_embedding
        self.reduction = reduction
        self.normalize = normalize

    def forward(self, x):
        # x.shape = (sentence_length, batch_size)

        embedded = super().forward(x)
        # embedded.shape = (sentence_length, batch_size, embedding_size)

        # Reduce representation if specified to mean of document word vector embeddings over sentence_length
        #   'mean' : mean of document word vector embeddings over sentence_length
        #   'max'  : max-pooling of document word vector embedding dimensions over sentence_length
        # After reduction: embedded.shape = (batch_size, embedding_size)
        if self.reduction != 'none':

            if self.reduction == 'mean':
                embedded = torch.mean(embedded, dim=0)

            if self.reduction == 'max':
                embedded, _ = torch.max(embedded, dim=0)

            if self.normalize:
                embedded = embedded / torch.norm(embedded, p=2, dim=1, keepdim=True).clamp(min=1e-08)
                embedded[torch.isnan(embedded)] = 0

        return embedded

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
