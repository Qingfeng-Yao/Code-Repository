import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import util

## 流模型
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

def create_degrees(input_dim,
                    hidden_dims,
                    input_order='left-to-right',
                    hidden_order='left-to-right'):
    degrees = []
    input_degrees = np.arange(1, input_dim + 1)
    if input_order == 'right-to-left':
        input_degrees = np.flip(input_degrees, 0)
    elif input_order == 'random':
        np.random.shuffle(input_degrees)
    
    degrees.append(input_degrees)

    for units in hidden_dims:
        if hidden_order == 'random':
            min_prev_degree = min(np.min(degrees[-1]), input_dim - 1)
            hidden_degrees = np.random.randint(
                    low=min_prev_degree, high=input_dim, size=units)
        elif hidden_order == 'left-to-right':
            hidden_degrees = (np.arange(units) % max(1, input_dim - 1) +
                                                min(1, input_dim - 1))
        degrees.append(hidden_degrees)
    return degrees

def create_masks(input_dim,
                hidden_dims,
                input_order='left-to-right',
                hidden_order='left-to-right'):
    degrees = create_degrees(input_dim, hidden_dims, input_order, hidden_order)
    masks = []
    # Create input-to-hidden and hidden-to-hidden masks.
    for input_degrees, output_degrees in zip(degrees[:-1], degrees[1:]):
        mask = torch.Tensor(input_degrees[:, np.newaxis] <= output_degrees).float()
        masks.append(mask)

    # Create hidden-to-output mask.
    mask = torch.Tensor(degrees[-1][:, np.newaxis] < degrees[0]).float()
    masks.append(mask)
    return masks

class MaskedLinearDis(nn.Linear):
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask): 
        mask = mask.long().T
        self.mask.data.copy_(mask)
        # if all of the inputs are zero, need to ensure the bias is zeroed out!
        self.bias_all_zero_mask = (mask.sum(dim=1)!=0).float()
        
    def forward(self, input):
        self.bias_all_zero_mask = self.bias_all_zero_mask.to(input.device)
        return F.linear(input, self.mask * self.weight, self.bias_all_zero_mask * self.bias)

class MADE_dis(nn.Module):
    def __init__(self,
                input_shape, 
                units,
                hidden_dims,
                input_order='left-to-right',
                hidden_order='left-to-right',
                use_bias=True):
        super(MADE_dis, self).__init__()
        self.units = int(units)
        self.hidden_dims = hidden_dims
        self.input_order = input_order
        self.hidden_order = hidden_order
        self.use_bias = use_bias
        self.network = nn.ModuleList()
        self.build(input_shape)

    def build(self, input_shape):
        length = input_shape[-2]
        channels = input_shape[-1]
        masks = create_masks(input_dim=length,
                            hidden_dims=self.hidden_dims,
                            input_order=self.input_order,
                            hidden_order=self.hidden_order)

        # Input-to-hidden layer: [..., length, channels] -> [..., hidden_dims[0]]
        mask = masks[0]
        mask = mask.unsqueeze(1).repeat(1, channels, 1)
        mask = mask.view(mask.shape[0] * channels, mask.shape[-1])
        if self.hidden_dims:
            layer = MaskedLinearDis(channels*length, self.hidden_dims[0])
            layer.set_mask(mask)

            self.network.append(layer)
            self.network.append(nn.ReLU())

        # Hidden-to-hidden layers: [..., hidden_dims[l-1]] -> [..., hidden_dims[l]]
        for ind in range(1, len(self.hidden_dims)-1):
            layer = MaskedLinearDis(self.hidden_dims[ind], self.hidden_dims[ind+1])
            layer.set_mask(masks[ind])

            self.network.append(layer)
            self.network.append(nn.ReLU())

        # Hidden-to-output layer: [..., hidden_dims[-1]] -> [..., length, units].
        if self.hidden_dims:
            mask = masks[-1]
        mask = mask.unsqueeze(-1).repeat(1, 1, self.units)
        mask = mask.view(mask.shape[0], mask.shape[1] * self.units)
        layer = MaskedLinearDis(self.hidden_dims[-1],channels*length)
        layer.set_mask(mask)

        self.network.append(layer)
        self.network = nn.Sequential(*self.network)

    def forward(self, inputs):
        input_shapes = inputs.shape
        inputs = inputs.view(-1, input_shapes[-1]*input_shapes[-2])
        inputs = self.network(inputs)
        out = inputs.view(-1, input_shapes[-2], self.units)
        return out

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

class MAF(nn.Module):
    def __init__(self, flows):
        super().__init__()

        self.flows = flows

    def forward(self, inputs, cond_inputs = None):

        return -self.flows.log_probs(inputs, cond_inputs)

class DiscreteAutoregressiveFlow(nn.Module):
    def __init__(self, layer, temperature, vocab_size):
        super().__init__()
        self.layer = layer
        self.temperature = temperature
        self.vocab_size = vocab_size

    def forward(self, inputs):
        net = self.layer(inputs)
        if net.shape[-1] == 2 * self.vocab_size:
            loc, scale = torch.split(net, self.vocab_size, dim=-1)
            scale = util.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            scaled_inputs = util.one_hot_multiply(inputs, scale)
        elif net.shape[-1] == self.vocab_size:
            loc = net
            scaled_inputs = inputs
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        loc = util.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
        outputs = util.one_hot_add(scaled_inputs, loc)
        return outputs

class DiscreteAutoFlowModel(nn.Module):
    # combines all of the discrete flow layers into a single model
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
         # from the data to the latent space
        for flow in self.flows:
            z = flow.forward(z)
        return z

    def reverse(self, x):
        # from the latent space to the data
        for flow in self.flows[::-1]:
            x = flow.reverse(x)
        return x

## 语言模型