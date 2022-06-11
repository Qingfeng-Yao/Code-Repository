import torch
import torch.nn as nn 
import numpy as np 
import sys
sys.path.append("../../")

from embedflow.general.mutils import get_param_val, create_transformer_mask, create_channel_mask
from embedflow.layers.flows.flow_model import FlowModel 
from embedflow.layers.flows.activation_normalization import ActNormFlow
from embedflow.layers.flows.permutation_layers import InvertibleConv
from embedflow.layers.flows.autoregressive_coupling import AutoregressiveMixtureCDFCoupling
from embedflow.layers.networks.autoregressive_layers import AutoregressiveLSTMModel
from embedflow.layers.categorical_encoding.mutils import create_encoding
from embedflow.layers.categorical_encoding.decoder import create_embed_layer


class FlowLanguageModeling(FlowModel):


	def __init__(self, model_params, dataset_class, vocab_size, vocab, emb):
		super().__init__(layers=None, name="Language Modeling Flow")
		self.model_params = model_params
		self.dataset_class = dataset_class
		self.max_seq_len = self.model_params["max_seq_len"]
		self.vocab_size = vocab_size
		self.vocab = vocab
		self.embed_layer = emb

		self._create_layers()
		self.print_overview()


	def _create_layers(self):

		self.latent_dim = self.model_params["categ_encoding"]["num_dimensions"]
		model_func = lambda c_out : AutoregressiveLSTMModel(c_in=self.latent_dim,
															c_out=c_out,
															max_seq_len=self.max_seq_len,
															num_layers=self.model_params["coupling_hidden_layers"],
															hidden_size=self.model_params["coupling_hidden_size"],
															dp_rate=self.model_params["coupling_dropout"],
															input_dp_rate=self.model_params["coupling_input_dropout"])
		self.model_params["categ_encoding"]["flow_config"]["model_func"] = model_func
		# self.encoding_layer = create_encoding(self.model_params["categ_encoding"], 
		# 									  dataset_class=self.dataset_class, 
		# 									  vocab_size=self.vocab_size,
		# 									  vocab=self.vocab)

		num_flows = self.model_params["coupling_num_flows"]

		layers = []
		for flow_index in range(num_flows):
			layers += [ActNormFlow(self.latent_dim)]
			if flow_index > 0:
				layers += [InvertibleConv(self.latent_dim)]
			layers += [
				AutoregressiveMixtureCDFCoupling(
						c_in=self.latent_dim,
						model_func=model_func,
						block_type="LSTM model",
						num_mixtures=self.model_params["coupling_num_mixtures"])
			]

		# self.flow_layers = nn.ModuleList([self.encoding_layer] + layers)
		self.flow_layers = nn.ModuleList(layers)


	def forward(self, z, ldj=None, reverse=False, length=None, **kwargs):
		if length is not None:
			kwargs["src_key_padding_mask"] = create_transformer_mask(length)
			kwargs["channel_padding_mask"] = create_channel_mask(length)

		batch_size, seq_length = z.size(0), z.size(1)
		z = z.reshape((batch_size * seq_length, 1) + z.shape[2:])
		if kwargs["channel_padding_mask"] is not None:
			channel_padding_mask = kwargs["channel_padding_mask"].reshape(batch_size * seq_length, 1, -1)
		else:
			channel_padding_mask = z.new_ones((batch_size * seq_length, 1, 1), dtype=torch.float32)
		
		z_categ = z
		z_cont = self.embed_layer(z_categ)
		# z_cont = self.embed_layer_2(z_cont)

		z_cont = z_cont * channel_padding_mask
		z_out = z_cont
		z = z_out.reshape(batch_size, seq_length, -1)

		return super().forward(z, ldj=ldj, reverse=reverse, length=length, **kwargs)


	def initialize_data_dependent(self, batch_list):
		# Batch list needs to consist of tuples: (z, kwargs)
		print("Initializing data dependent...")
		with torch.no_grad():
			new_batch_list = []
			for batch, kwargs in batch_list:
				kwargs["src_key_padding_mask"] = create_transformer_mask(kwargs["length"])
				kwargs["channel_padding_mask"] = create_channel_mask(kwargs["length"])

				z = batch
				batch_size, seq_length = z.size(0), z.size(1)
				z = z.reshape((batch_size * seq_length, 1) + z.shape[2:])

				channel_padding_mask = kwargs["channel_padding_mask"].reshape(batch_size * seq_length, 1, -1)
				
				z_categ = z
				z_cont = self.embed_layer(z_categ)
				# z_cont = self.embed_layer_2(z_cont)

				z_cont = z_cont * channel_padding_mask
				z_out = z_cont
				z = z_out.reshape(batch_size, seq_length, -1)
				new_batch_list.append((z, kwargs))

			for layer_index, layer in enumerate(self.flow_layers):
				batch_list = FlowModel.run_data_init_layer(new_batch_list, layer)


