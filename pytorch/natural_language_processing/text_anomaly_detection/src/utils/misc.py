from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import string
import re
import torch
import sys

import torch.nn as nn
import torch.nn.functional as F

def _create_length_mask(length, max_len=None, dtype=torch.float32):
    if max_len is None:
        max_len = length.max()
    mask = (torch.arange(max_len, device=length.device).view(1, max_len) < length.unsqueeze(dim=-1)).to(dtype=dtype)
    return mask

def create_transformer_mask(length, max_len=None, dtype=torch.float32):
	mask = _create_length_mask(length=length, max_len=max_len, dtype=torch.bool)
	mask = ~mask # Negating mask, as positions that should be masked, need a True, and others False
	# mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask

def create_channel_mask(length, max_len=None, dtype=torch.float32):
	mask = _create_length_mask(length=length, max_len=max_len, dtype=dtype)
	mask = mask.unsqueeze(dim=-1) # Unsqueeze over channels
	return mask

def create_T_one_hot(length, dataset_max_len, dtype=torch.float32):
    if length is None:
        print("Length", length)
        print("Dataset max len", dataset_max_len)
    max_batch_len = length.max()
    assert max_batch_len <= dataset_max_len, "[!] ERROR - T_one_hot: Max batch size (%s) was larger than given dataset max length (%s)" % (str(max_batch_len.item()), str(dataset_max_len))
    time_range = torch.arange(max_batch_len, device=length.device).view(1, max_batch_len).expand(length.size(0),-1)
    length_onehot_pos = one_hot(x=time_range, num_classes=dataset_max_len, dtype=dtype)
    inv_time_range = (length.unsqueeze(dim=-1)-1) - time_range
    length_mask = (inv_time_range>=0.0).float()
    inv_time_range = inv_time_range.clamp(min=0.0)
    length_onehot_neg = one_hot(x=inv_time_range, num_classes=dataset_max_len, dtype=dtype)
    length_onehot = torch.cat([length_onehot_pos, length_onehot_neg], dim=-1)
    length_onehot = length_onehot * length_mask.unsqueeze(dim=-1)
    return length_onehot

def one_hot(x, num_classes, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x_onehot = np.zeros(x.shape + (num_classes,), dtype=np.float32)
        x_onehot[np.arange(x.shape[0]), x] = 1.0
    elif isinstance(x, torch.Tensor):
        assert torch.max(x) < num_classes, "[!] ERROR: One-hot input has larger entries (%s) than classes (%i)" % (str(torch.max(x)), num_classes)
        x_onehot = x.new_zeros(x.shape + (num_classes,), dtype=dtype)
        x_onehot.scatter_(-1, x.unsqueeze(dim=-1), 1)
    else:
        print("[!] ERROR: Unknown object given for one-hot conversion:", x)
        sys.exit(1)
    return x_onehot

def run_padded_LSTM(x, lstm_cell, length, input_memory=None, return_final_states=False):
    if length is not None and (length != x.size(1)).sum() > 0:
        # Sort input elements for efficient LSTM application
        sorted_lengths, perm_index = length.sort(0, descending=True)
        x = x[perm_index]

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, sorted_lengths.cpu(), batch_first=True)
        packed_outputs, _ = lstm_cell(packed_input, input_memory)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # Redo sort
        _, unsort_indices = perm_index.sort(0, descending=False)
        outputs = outputs[unsort_indices]
    else:
        outputs, _ = lstm_cell(x, input_memory)
    return outputs

def run_sequential_with_mask(net, x, length=None, channel_padding_mask=None, src_key_padding_mask=None, length_one_hot=None, time_embed=None, gt=None, importance_weight=1, detail_out=False, **kwargs):
    dict_detail_out = dict()
    if channel_padding_mask is None:
        nn_out = net(x)
    else:
        x = x * channel_padding_mask
        for l in net:
            x = l(x)
        nn_out = x * channel_padding_mask # Making sure to zero out the outputs for all padding symbols

    if not detail_out:
        return nn_out
    else:
        return nn_out, dict_detail_out

def create_embed_layer(vocab_size, vocab, word_vectors, default_embed_layer_dims):
    if word_vectors is not None:
        embed_layer_dims = word_vectors.vectors.shape[1]
    else:
        embed_layer_dims = default_embed_layer_dims
    embed_layer = nn.Embedding(vocab_size, embed_layer_dims)
    if word_vectors is not None:
        for i, token in enumerate(vocab):
            embed_layer.weight.data[i] = word_vectors[token]
    embed_layer.weight.requires_grad = True
    return embed_layer, vocab_size

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

def _log_cdf(x, mean, log_scale):
	"""Element-wise log CDF of the logistic distribution."""
	z = (x - mean) * torch.exp(-log_scale)
	log_p = F.logsigmoid(z)

	return log_p

def _log_pdf(x, mean, log_scale):
	"""Element-wise log density of the logistic distribution."""
	z = (x - mean) * torch.exp(-log_scale)
	log_p = z - log_scale - 2 * F.softplus(z)

	return log_p

def mixture_log_cdf(x, prior_logits, means, log_scales):
	"""Log CDF of a mixture of logistic distributions."""
	log_ps = F.log_softmax(prior_logits, dim=-1) \
		+ _log_cdf(x.unsqueeze(dim=-1), means, log_scales)
	log_p = torch.logsumexp(log_ps, dim=-1)

	return log_p

def safe_log(x):
	return torch.log(x.clamp(min=1e-22))

def inverse_func(x, reverse=False):
	"""Inverse logistic function."""
	if reverse:
		z = torch.sigmoid(x)
		ldj = F.softplus(x) + F.softplus(-x)
	else:
		z = -safe_log(x.reciprocal() - 1.)
		ldj = -safe_log(x) - safe_log(1. - x)

	return z, ldj

def mixture_log_pdf(x, prior_logits, means, log_scales):
	"""Log PDF of a mixture of logistic distributions."""
	log_ps = F.log_softmax(prior_logits, dim=-1) \
		+ _log_pdf(x.unsqueeze(dim=-1), means, log_scales)
	log_p = torch.logsumexp(log_ps, dim=-1)

	return log_p

def mixture_inv_cdf(y, prior_logits, means, log_scales,
            		eps=1e-10, max_iters=100):
	# Inverse CDF of a mixture of logisitics. Iterative algorithm.
	if y.min() <= 0 or y.max() >= 1:
		raise RuntimeError('Inverse logisitic CDF got y outside (0, 1)')

	def body(x_, lb_, ub_):
		cur_y = torch.exp(mixture_log_cdf(x_, prior_logits, means,
		                                  log_scales))
		gt = (cur_y > y).type(y.dtype)
		lt = 1 - gt
		new_x_ = gt * (x_ + lb_) / 2. + lt * (x_ + ub_) / 2.
		new_lb = gt * lb_ + lt * x_
		new_ub = gt * x_ + lt * ub_
		return new_x_, new_lb, new_ub

	x = torch.zeros_like(y)
	max_scales = torch.sum(torch.exp(log_scales), dim=-1, keepdim=True)
	lb, _ = (means - 20 * max_scales).min(dim=-1)
	ub, _ = (means + 20 * max_scales).max(dim=-1)
	diff = float('inf')

	i = 0
	while diff > eps and i < max_iters:
		new_x, lb, ub = body(x, lb, ub)
		diff = (new_x - x).abs().max()
		x = new_x
		i += 1

	return x