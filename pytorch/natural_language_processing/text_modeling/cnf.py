import torch
import torch.nn as nn
import torch.nn.functional as F 

from layers import TimeConcat, ActNormFlow, InvertibleConv, AutoregressiveMixtureCDFCoupling, create_encoding
from utils import run_padded_LSTM,  create_transformer_mask, create_channel_mask

class FlowModel(nn.Module):
	
	def __init__(self, layers=None, name="Flow model"):
		super().__init__()

		self.flow_layers = nn.ModuleList()
		self.name = name

		if layers is not None:
			self.add_layers(layers)


	def add_layers(self, layers):
		for l in layers:
			self.flow_layers.append(l)
		self.print_overview()

	def forward(self, z, ldj=None, reverse=False, get_ldj_per_layer=False, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0), dtype=torch.float32)

		ldj_per_layer = []
		for layer_index, layer in (enumerate(self.flow_layers) if not reverse else reversed(list(enumerate(self.flow_layers)))):
			
			layer_res = layer(z, reverse=reverse, get_ldj_per_layer=get_ldj_per_layer, **kwargs)

			if len(layer_res) == 2:
				z, layer_ldj = layer_res 
				detailed_layer_ldj = layer_ldj
			elif len(layer_res) == 3:
				z, layer_ldj, detailed_layer_ldj = layer_res
			else:
				print("[!] ERROR: Got more return values than expected: %i" % (len(layer_res)))

			assert torch.isnan(z).sum() == 0, "[!] ERROR: Found NaN latent values. Layer (%i):\n%s" % (layer_index + 1, layer.info())
			
			ldj = ldj + layer_ldj
			if isinstance(detailed_layer_ldj, list):
				ldj_per_layer += detailed_layer_ldj
			else:
				ldj_per_layer.append(detailed_layer_ldj)

		if get_ldj_per_layer:
			return z, ldj, ldj_per_layer
		else:
			return z, ldj

	def reverse(self, z):
		return self.forward(z, reverse=True)

	def test_reversibility(self, z, **kwargs):
		test_failed = False
		for layer_index, layer in enumerate(self.flow_layers):
			z_layer, ldj_layer = layer(z, reverse=False, **kwargs)
			z_reconst, ldj_reconst = layer(z_layer, reverse=True, **kwargs)

			ldj_diff = (ldj_layer + ldj_reconst).abs().sum()
			z_diff = (z_layer - z_reconst).abs().sum()

			if z_diff != 0 or ldj_diff != 0:
				print("-"*100)
				print("[!] WARNING: Reversibility check failed for layer index %i" % layer_index)
				print(layer.info())
				print("-"*100)
				test_failed = True

		print("+"*100)
		print("Reversibility test %s (tested %i layers)" % ("failed" if test_failed else "succeeded", len(self.flow_layers)))
		print("+"*100)

	def get_inner_activations(self, z, reverse=False, return_names=False, **kwargs):
		out_per_layer = [z.detach()]
		layer_names = []
		for layer_index, layer in enumerate((self.flow_layers if not reverse else reversed(self.flow_layers))):
			
			z = layer(z, reverse=reverse, **kwargs)[0]
			out_per_layer.append(z.detach())
			layer_names.append(layer.__class__.__name__)

		if not return_names:
			return out_per_layer
		else:
			return out_per_layer, return_names

	def initialize_data_dependent(self, batch_list):
		# Batch list needs to consist of tuples: (z, kwargs)
		with torch.no_grad():
			for layer_index, layer in enumerate(self.flow_layers):
				print("Processing layer %i..." % (layer_index+1), end="\r")
				batch_list = FlowModel.run_data_init_layer(batch_list, layer)

	@staticmethod
	def run_data_init_layer(batch_list, layer):
		if layer.need_data_init():
			stacked_kwargs = {key: [b[1][key] for b in batch_list] for key in batch_list[0][1].keys()}
			for key in stacked_kwargs.keys():
				if isinstance(stacked_kwargs[key][0], torch.Tensor):
					stacked_kwargs[key] = torch.cat(stacked_kwargs[key], dim=0)
				else:
					stacked_kwargs[key] = stacked_kwargs[key][0]
			if not (isinstance(batch_list[0][0], tuple) or isinstance(batch_list[0][0], list)):
				input_data = torch.cat([z for z, _ in batch_list], dim=0)
				layer.data_init_forward(input_data, **stacked_kwargs)
			else:
				input_data = [torch.cat([z[i] for z, _ in batch_list], dim=0) for i in range(len(batch_list[0][0]))]
				layer.data_init_forward(*input_data, **stacked_kwargs)
		out_list = []
		for z, kwargs in batch_list:
			if isinstance(z, tuple) or isinstance(z, list):
				z = layer(*z, reverse=False, **kwargs)
				out_list.append([e.detach() for e in z[:-1] if isinstance(e, torch.Tensor)])
				if len(z) == 4 and isinstance(z[-1], dict):
					kwargs.update(z[-1])
					out_list[-1] = out_list[-1][:-1]
			else:
				z = layer(z, reverse=False, **kwargs)[0]
				out_list.append(z.detach())
		batch_list = [(out_list[i], batch_list[i][1]) for i in range(len(batch_list))]
		return batch_list

	def need_data_init(self):
		return any([flow.need_data_init() for flow in self.flow_layers])

	def print_overview(self):
		# Retrieve layer descriptions for all flows
		layer_descp = list()
		for layer_index, layer in enumerate(self.flow_layers):
			layer_descp.append("(%2i) %s" % (layer_index+1, layer.info()))
		num_tokens = max([20] + [len(s) for s in "\n".join(layer_descp).split("\n")])
		# Print out info in a nicer format
		print("="*num_tokens)
		print("%s with %i flows" % (self.name, len(self.flow_layers)))
		print("-"*num_tokens)
		print("\n".join(layer_descp))
		print("="*num_tokens)

class AutoregFeedforward(nn.Module):
	
	def __init__ (self, c_in, c_out_per_in, hidden_size, c_offset=0):
		super().__init__()
		self.c_in = c_in
		self.c_autoreg = c_in - 1 - c_offset
		self.c_out_per_in = c_out_per_in
		self.c_offset = c_offset
		self.hidden_size = hidden_size
		self.embed_size = min(max(1, int(hidden_size*9.0/16.0/(self.c_in-1))), 96)
		self.hidden_dim_2 = int(hidden_size//2)
		self.act_fn_1 = nn.GELU()
		self.act_fn_2 = nn.GELU()
		self.in_to_features = nn.Linear((self.c_in-1)*3, self.embed_size*(self.c_in-1))
		self.features_to_hidden = nn.Linear(hidden_size + self.embed_size*(self.c_in-1), self.hidden_dim_2*self.c_in)
		self.hidden_to_out = nn.Linear(self.hidden_dim_2*self.c_in, c_out_per_in*self.c_in)
		mask_in_to_features, mask_features_to_hidden, mask_hidden_to_out = self._create_masks()
		self.register_buffer("mask_in_to_features", mask_in_to_features)
		self.register_buffer("mask_features_to_hidden", mask_features_to_hidden)
		self.register_buffer("mask_hidden_to_out", mask_hidden_to_out)

	def forward(self, features, _inps):
		self._mask_layers()
		if _inps.size(-1) == self.c_in:
			_inps = _inps[...,:-1] # The last channel is not used as input for any transformation
		_inps = torch.stack([_inps, F.elu(_inps), F.elu(-_inps)], dim=-1).view(_inps.shape[:-1]+(3*_inps.shape[-1],))
		in_features = self.in_to_features(_inps)
		in_features = self.act_fn_1(in_features)
		features = torch.cat([features, in_features], dim=-1)
		hidden = self.features_to_hidden(features)
		hidden = self.act_fn_2(hidden)
		out = self.hidden_to_out(hidden)
		return out

	def _create_masks(self):
		mask_in_to_features = torch.ones_like(self.in_to_features.weight.data) # [self.embed_size*(c_in-1), c_in-1]
		for c_in in range(self.c_offset, self.c_in-1):
			mask_in_to_features[:self.embed_size*c_in, c_in*3:(c_in+1)*3] = 0
			mask_in_to_features[self.embed_size*(c_in+1):, c_in*3:(c_in+1)*3] = 0
			mask_in_to_features[self.embed_size*c_in:self.embed_size*(c_in+1), :c_in*3] = 0
			mask_in_to_features[self.embed_size*c_in:self.embed_size*(c_in+1), (c_in+1)*3:] = 0

		mask_features_to_hidden = torch.ones_like(self.features_to_hidden.weight.data) # [self.hidden_dim_2*c_in, hidden_size + self.embed_size*(c_in-1)]
		for c_in in range(self.c_in):
			mask_features_to_hidden[self.hidden_dim_2*c_in:self.hidden_dim_2*(c_in+1), self.hidden_size+self.embed_size*(self.c_offset + max(0,c_in-self.c_offset)):] = 0
		
		mask_hidden_to_out = torch.ones_like(self.hidden_to_out.weight.data) # [c_out_per_in*c_in, self.hidden_dim_2*c_in]
		for c_in in range(self.c_in):
			mask_hidden_to_out[:self.c_out_per_in*c_in, self.hidden_dim_2*c_in:self.hidden_dim_2*(c_in+1)] = 0
			mask_hidden_to_out[self.c_out_per_in*(c_in+1):, self.hidden_dim_2*c_in:self.hidden_dim_2*(c_in+1)] = 0

		return mask_in_to_features, mask_features_to_hidden, mask_hidden_to_out

	def _mask_layers(self):
		self.in_to_features.weight.data *= self.mask_in_to_features
		self.features_to_hidden.weight.data *= self.mask_features_to_hidden
		self.hidden_to_out.weight.data *= self.mask_hidden_to_out

class LSTMFeatureModel(nn.Module):
	
	def __init__(self, c_in, c_out, hidden_size, max_seq_len,
					   num_layers=1, dp_rate=0.0, 
					   input_dp_rate=0.0, **kwargs):
		super().__init__()

		time_embed = nn.Linear(2*max_seq_len, int(hidden_size//8))
		time_embed_dim = time_embed.weight.data.shape[0]
		self.time_concat = TimeConcat(time_embed=time_embed, input_dp_rate=input_dp_rate)
		inp_embed_dim = hidden_size//2 - time_embed_dim
		self.input_embed = nn.Sequential(
				nn.Linear(c_in, hidden_size//2),
				nn.GELU(),
				nn.Linear(hidden_size//2, inp_embed_dim),
				nn.GELU()
			)

		self.lstm_module = nn.LSTM(input_size=inp_embed_dim + time_embed_dim, hidden_size=hidden_size,
									num_layers=num_layers, batch_first=True,
									bidirectional=False, dropout=0.0)
		self.out_layer = AutoregFeedforward(c_in=c_in, c_out_per_in=int(c_out/c_in), 
									   hidden_size=hidden_size//2, c_offset=0)
		self.net = nn.Sequential(
				nn.Dropout(dp_rate),
				nn.Linear(hidden_size, hidden_size//2),
				nn.GELU(),
				nn.Dropout(dp_rate)
			)

	def forward(self, x, length=None, channel_padding_mask=None, length_one_hot=None, **kwargs):
		_inp_embed = self.input_embed(x)
		embed = self.time_concat(x=_inp_embed, length_one_hot=length_one_hot, length=length)
		embed = torch.cat([embed.new_zeros(embed.size(0),1,embed.size(2)), embed[:,:-1]], dim=1)

		lstm_out = run_padded_LSTM(x=embed, lstm_cell=self.lstm_module, length=length)

		out = self.net(lstm_out)
		out = self.out_layer(features=out, _inps=x)
		if channel_padding_mask is not None:
			out = out * channel_padding_mask
		return out

class AutoregressiveLSTMModel(nn.Module):
	
	def __init__(self, c_in, c_out, hidden_size, max_seq_len, 
					   num_layers=1, dp_rate=0.0, 
					   input_dp_rate=0.0,
					   direction=0, **kwargs):
		super().__init__()
		self.lstm_model = LSTMFeatureModel(c_in, c_out, hidden_size, num_layers=num_layers,
						max_seq_len=max_seq_len,
						dp_rate=dp_rate, input_dp_rate=input_dp_rate)
		self.reverse = (direction == 1)

	def forward(self, x, length=None, channel_padding_mask=None, **kwargs):
		if self.reverse:
			x = self._reverse_input(x, length, channel_padding_mask)
		x = self.lstm_model(x, length=length, channel_padding_mask=channel_padding_mask, shift_by_one=True, **kwargs)
		if self.reverse:
			x = self._reverse_input(x, length, channel_padding_mask)
		return x

	def _reverse_input(self, x, length, channel_padding_mask):
		max_batch_len = x.size(1)
		time_range = torch.arange(max_batch_len, device=length.device).view(1, max_batch_len).expand(length.size(0),-1).long()
		indices = ((length.unsqueeze(dim=1)-1) - time_range).clamp(min=0)
		indices = indices.unsqueeze(dim=-1).expand(-1, -1, x.size(2))
		x_inv = x.gather(index=indices, dim=1)
		x_inv = x_inv * channel_padding_mask
		return x_inv

class CNFLanguageModeling(FlowModel):

	def __init__(self, model_params, dataset_class, vocab_size, vocab):
		super().__init__(layers=None, name="Language Modeling Flow")
		self.model_params = model_params
		self.dataset_class = dataset_class
		self.max_seq_len = self.model_params["max_seq_len"]
		self.vocab_size = vocab_size
		self.vocab = vocab

		self._create_layers()
		self.print_overview()


	def _create_layers(self):
		self.latent_dim = self.model_params["categ_encoding"]["num_dimensions"]
		model_func = lambda c_out : AutoregressiveLSTMModel(c_in=self.latent_dim, c_out=c_out, max_seq_len=self.max_seq_len, num_layers=self.model_params["coupling_hidden_layers"], hidden_size=self.model_params["coupling_hidden_size"], dp_rate=self.model_params["coupling_dropout"], input_dp_rate=self.model_params["coupling_input_dropout"])
		self.model_params["categ_encoding"]["flow_config"]["model_func"] = model_func
		self.encoding_layer = create_encoding(self.model_params["categ_encoding"], dataset_class=self.dataset_class, vocab_size=self.vocab_size, vocab=self.vocab)

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

		self.flow_layers = nn.ModuleList([self.encoding_layer] + layers)


	def forward(self, z, ldj=None, reverse=False, length=None, **kwargs):
		if length is not None:
			kwargs["src_key_padding_mask"] = create_transformer_mask(length)
			kwargs["channel_padding_mask"] = create_channel_mask(length)
		return super().forward(z, ldj=ldj, reverse=reverse, length=length, **kwargs)


	def initialize_data_dependent(self, batch_list):
		# Batch list needs to consist of tuples: (z, kwargs)
		print("Initializing data dependent...")
		with torch.no_grad():
			for batch, kwargs in batch_list:
				kwargs["src_key_padding_mask"] = create_transformer_mask(kwargs["length"])
				kwargs["channel_padding_mask"] = create_channel_mask(kwargs["length"])
			for layer_index, layer in enumerate(self.flow_layers):
				batch_list = FlowModel.run_data_init_layer(batch_list, layer)

