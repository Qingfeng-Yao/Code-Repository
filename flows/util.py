import torch
import torch.nn.functional as F
import numpy as np

## discrete utils
def one_hot(inputs, vocab_size = None):
    """Returns one hot of data over each element of the inputs"""
    if vocab_size is None:
        vocab_size = inputs.max() + 1
    input_shape = inputs.shape
    inputs = inputs.flatten().unsqueeze(1).long()
    z = torch.zeros(len(inputs), vocab_size).to(inputs.device)
    z.scatter_(1, inputs, 1.)
    return z.view(*input_shape, vocab_size)

def one_hot_argmax(inputs, temperature, axis=-1):
    """Returns one-hot of argmax with backward pass set to softmax-temperature."""
    vocab_size = inputs.shape[-1]
    z = one_hot(torch.argmax(inputs, dim=axis), vocab_size) 
    soft = F.softmax(inputs / temperature, dim=axis)
    outputs = soft + (z - soft).detach()
    return outputs

def one_hot_multiply(inputs, scale):
    scale = scale.type(inputs.dtype)
    batch_shape = list(inputs.shape[:-1])
    vocab_size = inputs.shape[-1]

    to_perm = torch.arange(vocab_size).unsqueeze(1).repeat(1, vocab_size) * torch.arange(vocab_size).unsqueeze(0)
    permutation_matrix = one_hot(torch.fmod(to_perm,vocab_size))

    scaled_inputs = torch.einsum('...v,avu->...au', inputs, permutation_matrix)
    scaled_inputs = torch.cat( (torch.zeros(batch_shape + [1, vocab_size]),
                                scaled_inputs[..., 1:, :]), dim=-2)

    outputs = torch.einsum('...v,...vu->...u', scale, scaled_inputs)
    return outputs

def one_hot_add(inputs, shift):
    inputs = torch.stack((inputs, torch.zeros_like(inputs)), dim = -1)
    shift = torch.stack((shift, torch.zeros_like(shift)), dim = -1)
    inputs_fft = torch.fft(inputs, 1) #ignore last and first dimension to do batched fft
    shift_fft = torch.fft(shift, 1)
    result_fft_real = inputs_fft[...,0]*shift_fft[...,0] - inputs_fft[...,1]*shift_fft[...,1]
    result_fft_imag = inputs_fft[...,0]*shift_fft[...,1] + inputs_fft[...,1]*shift_fft[...,0]
    result_fft = torch.stack((result_fft_real,result_fft_imag), dim = -1)
    return torch.ifft(result_fft, 1)[...,0]