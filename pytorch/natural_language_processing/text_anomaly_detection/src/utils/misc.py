from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import string
import re
import torch
import sys

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

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)
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
