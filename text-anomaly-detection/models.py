import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel

import util

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

## -----------------


## -----------------
## 流模型相关
def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output
nn.MaskedLinear = MaskedLinear
    
class MADE(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 n_hidden,
                 num_cond_inputs=None,
                 act='relu'):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_masks = []
        for _ in range(n_hidden):
            hidden_masks.append(get_mask(num_hidden, num_hidden, num_inputs))
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = []
        for m in hidden_masks:
            self.trunk += [act_func(), nn.MaskedLinear(num_hidden, num_hidden,
                                                   m)]
        self.trunk += [act_func(), nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask)]
        self.trunk = nn.Sequential(*self.trunk)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, -1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[-1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, -1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)

class BatchNormFlow(nn.Module):
    def __init__(self, num_inputs, momentum=0.9, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.reshape(-1, inputs.shape[-1]).mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).reshape(-1, inputs.shape[-1]).mean(0)

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / (var+self.eps).sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var + self.eps)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * (var+self.eps).sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var+self.eps)).sum(
                -1, keepdim=True)

class Reverse(nn.Module):
    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, :, self.perm], torch.zeros(
                inputs.size(0), inputs.size(1), 1, device=inputs.device)
        else:
            return inputs[:, :, self.inv_perm], torch.zeros(
                inputs.size(0), inputs.size(1), 1, device=inputs.device)
 
class FlowSequential(nn.Sequential):
    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), inputs.size(1), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

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

## -----------------

## -----------------
## 模型集成相关
class ReduceTextFlowModel(nn.Module):
    def __init__(self, pretrained_model, flows):
        super().__init__()

        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size
        self.flows = flows

    def forward(self, x, pos=None, weights=None):
        # x.shape = (sentence_length, batch_size)
        # weights.shape = (sentence_length, batch_size)
        
        hidden = self.pretrained_model(x, weights)
        # hidden.shape = (batch_size, hidden_size)
        hidden = hidden.unsqueeze(1)

        log_probs = self.flows.log_probs(hidden)
        log_probs = torch.mean(log_probs, dim=1)

        return log_probs

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

class TempFlowModel(nn.Module):
    def __init__(self, pretrained_model, flows, cond_size=200):
        super().__init__()

        self.pretrained_model = pretrained_model
        self.embedding_size = pretrained_model.embedding_size
        self.hidden_size = 40
        self.num_layers = 2
        self.cond_size = cond_size

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.embedding_size, embedding_dim=self.embed_dim
        )

        self.rnn = nn.GRU(
            input_size=self.embedding_size * self.embed_dim+2*self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.1,
            batch_first=True,
        )
        self.out = nn.Linear(self.hidden_size, self.cond_size)

        self.flows = flows


    def forward(self, x, pos, weights=None):
        # x.shape = (sentence_length, batch_size)
        # pos.shape = (batch_size, sentence_length)

        pos_emb_model = nn.Embedding.from_pretrained(util.get_sinusoid_encoding_table(x.shape[0]+1, self.embedding_size),freeze=True).to(x.device)
        pos_emb = pos_emb_model(pos)
        # pos_emb = (batch_size, sentence_length, embedding_size)

        inputs = self.pretrained_model(x)
        # inputs.shape = (sentence_length, batch_size, embedding_size)

        sequences = inputs[:-1, :, :]
        sequences = sequences.permute(1, 0, 2)
        # sequences.shape = (batch_size, sentence_length-1, embedding_size)
        
        target_dimension_indicator = torch.arange(self.embedding_size).unsqueeze(0).repeat(x.shape[1],1).to(x.device)
        index_embeddings = self.embed(target_dimension_indicator)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, x.shape[0]-1, -1, -1)
            .reshape((-1, x.shape[0]-1, self.embedding_size * self.embed_dim))
        )
        # repeated_index_embeddings = (batch_size, sentence_length-1, embedding_size*embed_dim)

        outs = torch.cat((sequences, repeated_index_embeddings, pos_emb[:, 1: ,:]), dim=-1)
        # outs = (batch_size, sentence_length-1, 2*embedding_size+embedding_size*embed_dim)
        
        hidden = None

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(outs, hidden)
        # outputs : [batch_size, sentence_length-1, hidden_size]
        # hidden : [num_layers, batch_size, hidden_size]
      
        outputs = self.out(outputs)
        # outputs : [batch_size, sentence_length-1, cond_size]

        inputs += torch.rand_like(inputs).to(x.device)
        inputs = inputs.permute(1, 0, 2)

        likelihoods = self.flows.log_probs(inputs[:, 1:, :], outputs)
        # likelihoods : [batch_size, sentence_length-1, 1]
        log_probs = torch.mean(likelihoods, dim=1)

        return log_probs

class TransformerTempFlowModel(nn.Module):
    def __init__(self, pretrained_model, flows, cond_size=200):
        super().__init__()

        self.pretrained_model = pretrained_model
        self.embedding_size = pretrained_model.embedding_size
        self.d_model = 16
        self.num_heads = 4
        self.cond_size = cond_size
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.dim_feedforward_scale = 4

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.embedding_size, embedding_dim=self.embed_dim
        )

        self.encoder_input = nn.Linear(self.embedding_size * self.embed_dim+2*self.embedding_size, self.d_model)
        self.decoder_input = nn.Linear(self.embedding_size * self.embed_dim+2*self.embedding_size, self.d_model)

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward_scale * self.d_model,
            dropout=0.1,
            activation="gelu",
        )

        self.out = nn.Linear(self.d_model, self.cond_size)

        self.flows = flows


    def forward(self, x, pos, weights=None):
        # x.shape = (sentence_length, batch_size)
        # pos.shape = (batch_size, sentence_length)

        pos_emb_model = nn.Embedding.from_pretrained(util.get_sinusoid_encoding_table(x.shape[0]+1, self.embedding_size),freeze=True).to(x.device)
        pos_emb = pos_emb_model(pos)
        # pos_emb = (batch_size, sentence_length, embedding_size)

        inputs = self.pretrained_model(x)
        # inputs.shape = (sentence_length, batch_size, embedding_size)

        sequences = inputs[:-1, :, :]
        sequences = sequences.permute(1, 0, 2)
        # sequences.shape = (batch_size, sentence_length-1, embedding_size)
        
        target_dimension_indicator = torch.arange(self.embedding_size).unsqueeze(0).repeat(x.shape[1],1).to(x.device)
        index_embeddings = self.embed(target_dimension_indicator)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, x.shape[0]-1, -1, -1)
            .reshape((-1, x.shape[0]-1, self.embedding_size * self.embed_dim))
        )
        # repeated_index_embeddings = (batch_size, sentence_length-1, embedding_size*embed_dim)

        outs = torch.cat((sequences, repeated_index_embeddings, pos_emb[:, 1: ,:]), dim=-1)
        # outs = (batch_size, sentence_length-1, 2*embedding_size+embedding_size*embed_dim)

        enc_inputs = outs[:, :-1, :]
        dec_inputs = outs

        enc_out = self.transformer.encoder(
            self.encoder_input(enc_inputs).permute(1, 0, 2)
        )

        tgt_mask = self.transformer.generate_square_subsequent_mask(x.shape[0]-1).to(x.device)
        dec_output = self.transformer.decoder(
            self.decoder_input(dec_inputs).permute(1, 0, 2),
            enc_out,
            tgt_mask=tgt_mask,
        )

        outputs = dec_output.permute(1, 0, 2)
        outputs = self.out(outputs)
        # outputs : [batch_size, sentence_length-1, cond_size]

        inputs += torch.rand_like(inputs).to(x.device)
        inputs = inputs.permute(1, 0, 2)

        likelihoods = self.flows.log_probs(inputs[:, 1:, :], outputs)
        # likelihoods : [batch_size, sentence_length-1, 1]
        log_probs = torch.mean(likelihoods, dim=1)

        return log_probs

## -----------------