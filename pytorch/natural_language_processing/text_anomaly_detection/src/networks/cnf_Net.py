from base.base_net import BaseNet
from .cnf_flow_models import LinearCategoricalEncoding, AutoregressiveLSTMModel, ActNormFlow, InvertibleConv, AutoregressiveMixtureCDFCoupling
from utils.misc import create_transformer_mask, create_channel_mask

import torch
import torch.nn as nn

class CNFNet(BaseNet):
    
    def __init__(self, word_vectors=None, num_dimensions=3, dataset=None, encoding_params=None, coupling_hidden_size=1024, coupling_hidden_layers=2, coupling_num_flows=1, coupling_num_mixtures=64, coupling_dropout=0.0, coupling_input_dropout=0.0, max_seq_len=None, use_time_embed=False):
        super().__init__()
        self.word_vectors = word_vectors
        self.hidden_size = num_dimensions
        self.vocab_size = dataset.encoder.vocab_size
        self.vocab = dataset.encoder.vocab

        # Set normalization flow module
        self.coupling_hidden_size = coupling_hidden_size
        self.coupling_hidden_layers = coupling_hidden_layers
        self.coupling_num_flows = coupling_num_flows
        self.coupling_num_mixtures = coupling_num_mixtures
        self.coupling_dropout = coupling_dropout
        self.coupling_input_dropout = coupling_input_dropout
        self.max_seq_len = max_seq_len
        self.use_time_embed = use_time_embed

        self.encoding_layer = LinearCategoricalEncoding(num_dimensions=self.hidden_size, vocab_size=self.vocab_size, vocab=self.vocab, word_vectors=self.word_vectors, encoding_params=encoding_params)

        model_func = lambda c_out : AutoregressiveLSTMModel(c_in=self.hidden_size, c_out=c_out, max_seq_len=self.max_seq_len, num_layers=self.coupling_hidden_layers, hidden_size=self.coupling_hidden_size, dp_rate=self.coupling_dropout, input_dp_rate=self.coupling_input_dropout, use_time_embed=self.use_time_embed)
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
                        num_mixtures=self.coupling_num_mixtures)
            ]

        self.flow_layers = nn.ModuleList([self.encoding_layer]+layers)

    def forward(self, x, length, **kwargs):
        # x.shape = (sentence_length, batch_size)
        # length.shape = (batch_size, )

        # Change to hidden.shape = (batch_size, sentence_length)
        z = x.transpose(0, 1)

        kwargs["src_key_padding_mask"] = create_transformer_mask(length)
        kwargs["channel_padding_mask"] = create_channel_mask(length)
        kwargs["length"] = length

        ldj = z.new_zeros(z.size(0), dtype=torch.float32)
        for layer_index, layer in enumerate(self.flow_layers):
            layer_res = layer(z, reverse=False, get_ldj_per_layer=False, **kwargs)

            if len(layer_res) == 2:
                z, layer_ldj = layer_res
            elif len(layer_res) == 3:
                z, layer_ldj, _ = layer_res
            else:
                print("[!] ERROR: Got more return values than expected: %i" % (len(layer_res)))

            assert torch.isnan(z).sum() == 0, "[!] ERROR: Found NaN latent values. Layer (%i):\n%s" % (layer_index + 1, layer.info())
            
            ldj = ldj + layer_ldj

        return z, ldj