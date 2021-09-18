from base.base_net import BaseNet
from .flow_models import ActNormFlow, InvertibleConv
from utils.misc import create_transformer_mask, create_channel_mask

import torch
import torch.nn as nn

class ENFNet(BaseNet):
    
    def __init__(self, pretrained_model, coupling_num_flows=1):
        super().__init__()

        # Load pretrained model (which provides a hidden representation per word, e.g. word vector or language model)
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size

        # Set normalization flow module
        self.coupling_num_flows = coupling_num_flows
        layers = []
        for flow_index in range(self.coupling_num_flows):
            layers += [ActNormFlow(self.hidden_size)]
            layers += [InvertibleConv(self.hidden_size)]

        self.flow_layers = nn.ModuleList(layers)

    def forward(self, x, length, **kwargs):
        # x.shape = (sentence_length, batch_size)
        # length.shape = (batch_size, )

        hidden = self.pretrained_model(x)
        # hidden.shape = (sentence_length, batch_size, hidden_size)

        # Change to hidden.shape = (batch_size, sentence_length, hidden_size)
        hidden = hidden.transpose(0, 1)

        kwargs["src_key_padding_mask"] = create_transformer_mask(length)
        kwargs["channel_padding_mask"] = create_channel_mask(length)
        kwargs["length"] = length

        ldj = hidden.new_zeros(hidden.size(0), dtype=torch.float32)
        for layer_index, layer in enumerate(self.flow_layers):
            layer_res = layer(hidden, reverse=False, get_ldj_per_layer=False, **kwargs)

            if len(layer_res) == 2:
                z, layer_ldj = layer_res
            elif len(layer_res) == 3:
                z, layer_ldj, _ = layer_res
            else:
                print("[!] ERROR: Got more return values than expected: %i" % (len(layer_res)))

            assert torch.isnan(z).sum() == 0, "[!] ERROR: Found NaN latent values. Layer (%i):\n%s" % (layer_index + 1, layer.info())
            
            ldj = ldj + layer_ldj

        return z, ldj