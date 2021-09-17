from base.base_net import BaseNet
from .flow_models import AutoregressiveLSTMModel, ActNormFlow, InvertibleConv, AutoregressiveMixtureCDFCoupling
from utils.misc import create_transformer_mask, create_channel_mask

import torch
import torch.nn as nn

class ENFNet(BaseNet):
    
    def __init__(self, pretrained_model, coupling_hidden_size=1024, coupling_hidden_layers=2, coupling_num_flows=1, coupling_num_mixtures=64, coupling_dropout=0.0, coupling_input_dropout=0.0, max_seq_len=256):
        super().__init__()

        # Load pretrained model (which provides a hidden representation per word, e.g. word vector or language model)
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size

        # Set normalization flow module
        self.coupling_hidden_size = coupling_hidden_size
        self.coupling_hidden_layers = coupling_hidden_layers
        self.coupling_num_flows = coupling_num_flows
        self.coupling_num_mixtures = coupling_num_mixtures
        self.coupling_dropout= coupling_dropout
        self.coupling_input_dropout = coupling_input_dropout
        model_func = lambda c_out : AutoregressiveLSTMModel(c_in=self.hidden_size, c_out=c_out, max_seq_len=max_seq_len, num_layers=coupling_hidden_layers, hidden_size=coupling_hidden_size, dp_rate=coupling_dropout, input_dp_rate=coupling_input_dropout)
        layers = []
        for flow_index in range(self.coupling_num_flows):
            layers += [ActNormFlow(self.hidden_size)]
            if flow_index > 0:
                layers += [InvertibleConv(self.hidden_size)]
            layers += [
                AutoregressiveMixtureCDFCoupling(
                        c_in=self.hidden_size,
                        model_func=model_func,
                        block_type="LSTM model",
                        num_mixtures=coupling_num_mixtures)
            ]

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