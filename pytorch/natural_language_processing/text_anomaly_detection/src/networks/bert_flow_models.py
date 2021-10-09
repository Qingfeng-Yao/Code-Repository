import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True, logscale_factor=3., ddi=True):
        super().__init__()
        self.in_channel = in_channel

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.logscale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        torch.nn.init.xavier_uniform_(self.loc.data)
        torch.nn.init.xavier_uniform_(self.logscale.data)

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet
        self.logscale_factor = logscale_factor
        self.ddi = ddi

    def initialize(self, input):
        # input: (bsz, hdim, 1, 1)
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)  # (hdim, bsz)
            mean = flatten.mean(1)
            std = torch.sqrt(((flatten - mean.unsqueeze(-1)) ** 2).mean(dim=1))
            mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            self.loc.data.copy_(-mean)
            self.logscale.data.copy_(torch.log(1 / (std + 1e-6)) / self.logscale_factor)

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0 and self.ddi:
            self.initialize(input)
            self.initialized.fill_(1)

        logs = self.logscale * self.logscale_factor
        logdet = height * width * torch.sum(logs)
        output = torch.exp(logs) * (input + self.loc)

        if self.logdet:
            return output, logdet
        else:
            return output


# A random permutation
class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel

        weight = torch.zeros(in_channel, in_channel)
        for i in range(in_channel):
            weight[i, in_channel-1-i] = 1.
        weight = weight.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, input):
        out = F.conv2d(input, self.weight)
        return out, 0.0

class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.in_channel = in_channel

        self.conv = nn.Conv2d(in_channel, out_channel, [1, 1], padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)

        return out

class AdditiveCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=32):
        super().__init__()
        self.in_channel = in_channel

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, [1, 1], padding=0, bias=False),
            ActNorm(filter_size, logdet=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, [1, 1], padding=0, bias=False),
            ActNorm(filter_size, logdet=False),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel // 2),
        )

        # weight initialization
        for module in self.net:
            if type(module) != nn.Conv2d:
                continue
            module.weight.data.normal_(0, 0.05)

    def forward(self, x):
        x1, x2 = x.chunk(2, 1)
        z1 = x1
        shift = self.net(x1)
        z2 = x2 + shift
        output = torch.cat([z1, z2], dim=1)
        return output, 0.0

class Flow(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel

        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2d(in_channel)
        self.coupling = AdditiveCoupling(in_channel)

    def forward(self, x):
        objective = 0
        for fn in [self.actnorm, self.invconv, self.coupling]:
            x, obj = fn(x)
            objective += obj
        return x, objective

def compute_unconditional_prior(z):
    h = z.new_zeros(z.shape)
    prior_dist = torch.distributions.normal.Normal(h, torch.exp(h))
    return prior_dist

class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True):
        super().__init__()
        self.in_channel = in_channel

        self.flows = nn.ModuleList([Flow(in_channel) for _ in range(n_flow)])
        self.split = split

    def _get_eps(self, dist, x):
        return (x - dist.loc) / dist.scale
    
    def _set_eps(self, dist, eps):
        return eps * dist.scale + dist.loc

    def forward(self, x):
        b_size = x.shape[0]
        objective = 0

        for flow in self.flows:
            x, obj = flow(x)
            objective += obj

        eps = None
        if self.split:
            x1, x2 = x.chunk(2, 1)
            prior_dist = compute_unconditional_prior(x1)
            log_p = prior_dist.log_prob(x2). \
                        sum_to_size(b_size, 1, 1, 1). \
                        view(b_size)
            eps = self._get_eps(prior_dist, x2)
            x = x1
            objective = objective + log_p

        return x, objective, eps

class Glow(nn.Module):
    def __init__(self, in_channel, n_flow=3, n_block=2):
        super().__init__()
        self.in_channel = in_channel

        self.blocks = nn.ModuleList()
        for _ in range(n_block - 1):
            self.blocks.append(Block(in_channel, n_flow))
            in_channel //= 2
        self.blocks.append(Block(in_channel, n_flow, split=False))

    def forward(self, emb):
        # emb: (bsz, hdim)
        x = emb[:, :, None, None]  # b_size, n_channel, height, width
        b_size, c, h, w = x.shape
        
        log_p_sum = 0
        all_eps = []

        obj = 0
        for block in self.blocks:
            # print(x.shape)
            x, log_p, eps = block(x)
            if eps is not None:
                all_eps.append(eps)

            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        
        obj += log_p_sum
        z = x
        b_size = z.shape[0]
        prior_dist = compute_unconditional_prior(z)
        prior_objective = prior_dist.log_prob(z). \
                        sum_to_size(b_size, 1, 1, 1). \
                        view(b_size)
        if obj.shape != prior_objective.shape:
            obj = obj.unsqueeze(-1)
        obj = obj + prior_objective
        loss_batch = -obj / (np.log(2) * h * w * c)
        loss = (-obj / (np.log(2) * h * w * c)).mean()
        z = torch.cat(all_eps + [z], dim=1).view(b_size, c)
        return loss, loss_batch