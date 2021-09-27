from base.base_net import BaseNet
from .enf_flow_2D_models import MADE, BatchNormFlow, LUInvertibleMM, CouplingLayer, Reverse, FlowSequential
from utils.misc import create_transformer_mask, create_channel_mask

import torch
import torch.nn as nn

class ENFNet_2D(BaseNet):
    
    def __init__(self, pretrained_model, dataset, flow_type=None, coupling_hidden_size=1024, coupling_num_flows=1, use_length_prior=True, device='cuda'):
        super().__init__()

        # Load pretrained model (which provides a hidden representation per word, e.g. word vector or language model)
        self.pretrained_model = pretrained_model
        self.num_dims = pretrained_model.embedding_size
        self.use_length_prior = use_length_prior

        # Set normalization flow module
        self.hidden_size = coupling_hidden_size
        self.num_flows = coupling_num_flows

        modules = []
        if flow_type == 'maf':
            for _ in range(self.num_flows):
                modules += [
                MADE(self.num_dims, self.hidden_size, act='relu'),
                BatchNormFlow(self.num_dims),
                Reverse(self.num_dims)
            ]
        elif flow_type == 'glow':
            mask = torch.arange(0, self.num_dims) % 2
            mask = mask.to(device).float()
            for _ in range(self.num_flows):
                modules += [
                    BatchNormFlow(self.num_dims),
                    LUInvertibleMM(self.num_dims),
                    CouplingLayer(
                        self.num_dims, self.hidden_size, mask, s_act='tanh', t_act='relu')
                ]
                mask = 1 - mask

        self.flow_model = FlowSequential(*modules)

        if self.use_length_prior:
            self.length_prior = dataset.length_prior
        else:
            self.length_prior = None

    def forward(self, x, length):
        # x.shape = (sentence_length, batch_size)
        # length.shape = (batch_size, )

        hidden = self.pretrained_model(x)
        # hidden.shape = (sentence_length, batch_size, hidden_size)
        # or hidden.shape = (batch_size, hidden_size)

        if len(hidden.shape) == 3:
            # Change to hidden.shape = (batch_size, sentence_length, hidden_size)
            hidden = hidden.transpose(0, 1)

            channel_padding_mask = create_channel_mask(length)

            loss_mean, loss_batch  = self.flow_model.log_probs(hidden, length, self.length_prior, channel_padding_mask)
        else:
            loss_mean, loss_batch = self.flow_model.log_probs(hidden)

        return loss_mean, loss_batch