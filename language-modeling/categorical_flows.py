import torch
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np
from collections import defaultdict
import math
import scipy.linalg

import categorical_util

## LSTM model
class LSTMModel(nn.Module):

	def __init__(self, num_classes, hidden_size=64, num_layers=2, embedding_dim=32, dp_rate=0.0, input_dp_rate=0.0,
					   max_seq_len=-1, vocab=None, model_params=None):
		super().__init__()
		if model_params is not None:
			hidden_size = categorical_util.get_param_val(model_params, "coupling_hidden_size", hidden_size)
			embedding_dim = hidden_size//4
			num_layers = categorical_util.get_param_val(model_params, "coupling_hidden_layers", num_layers)
			dp_rate = categorical_util.get_param_val(model_params, "coupling_dropout", dp_rate)
			input_dp_rate = categorical_util.get_param_val(model_params, "coupling_input_dropout", input_dp_rate)
			max_seq_len = categorical_util.get_param_val(model_params, "max_seq_len", max_seq_len)

		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.embed_dim = 1 # Not equal to embedding_dim, is needed for making sampling equal to flows

		if vocab is not None and vocab.vectors is not None:
			embedding_dim = vocab.vectors.shape[1]
			self.embeddings = nn.Embedding(num_embeddings=len(vocab),
										   embedding_dim=embedding_dim)
			self.embeddings.weight.data.copy_(vocab.vectors)
			self.vocab_size = len(vocab)
		else:
			self.embeddings = nn.Embedding(num_embeddings=num_classes,
										   embedding_dim=embedding_dim)
			self.vocab_size = num_classes

		if input_dp_rate < 1.0:
			time_embed_dim = embedding_dim//4
			time_embed = nn.Linear(2*max_seq_len, time_embed_dim)
			self.max_seq_len = max_seq_len
			self.time_concat = TimeConcat(time_embed=time_embed, 
										  input_dp_rate=input_dp_rate)
		else:
			self.time_concat = None
			time_embed_dim = 0
		self.lstm = nn.LSTM(input_size=embedding_dim+time_embed_dim, 
							hidden_size=hidden_size, 
							num_layers=num_layers, 
							batch_first=True,
							bidirectional=False)

		self.init_state = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))

		self.output_layer = nn.Sequential(
				nn.Linear(hidden_size, hidden_size//2),
				nn.GELU(),
				nn.Dropout(dp_rate),
				nn.Linear(hidden_size//2, num_classes),
				nn.LogSoftmax(dim=-1)
			)


	def forward(self, x, reverse=False, **kwargs):
		if not reverse:
			return self._determine_logp(x, **kwargs)
		else:
			return self._sample(x.size(0), x.size(1))


	def _determine_logp(self, x, length=None, channel_padding_mask=None, **kwargs):
		if length is not None and channel_padding_mask is None:
			channel_padding_mask = categorical_util._create_length_mask(length, max_len=x.shape[1])
		x_embed = self.embeddings(x)
		if self.time_concat is not None:
			x_embed = self.time_concat(x=x_embed, length=length)
		init_state = self.init_state.expand(-1, x.size(0), -1).contiguous()
		h_0 = torch.tanh(init_state)
		c_0 = init_state

		x_prior = self.output_layer(h_0[0,:,:]) # Use initial state to predict prior
		x_hidden, (h_n, c_n) = self.lstm(x_embed, (h_0, c_0))
		x_pred = self.output_layer(x_hidden)
		logp = torch.cat([x_prior[:,None,:].gather(index=x[...,:1,None], dim=2),
						  x_pred[:,:-1,:].gather(index=x[...,1:,None], dim=2)], dim=1)

		logp = logp.sum(dim=2) # Shape: [Batch,Seq_len]
		if channel_padding_mask is not None:
			channel_padding_mask = channel_padding_mask.squeeze(dim=-1)
			logp = (logp * channel_padding_mask) / channel_padding_mask.sum(dim=[1], keepdims=True)

		ldj = logp.sum(dim=1)

		with torch.no_grad():
			detailed_log = {
				"avg_token_prob": -logp.exp().mean(dim=1),
				"avg_token_bpd": logp.mean(dim=1) * np.log2(np.exp(1))
			}

		return ldj, detailed_log


	def _sample(self, batch_size, seqlen):
		init_state = self.init_state.expand(-1, batch_size, -1).contiguous()
		h_0 = torch.tanh(init_state)
		c_0 = init_state

		preds = []
		x_prior = self.output_layer(h_0[0,:,:]).exp()
		preds.append(torch.multinomial(input=x_prior, num_samples=1).squeeze(dim=-1))
		h_n, c_n = h_0, c_0
		for _ in range(seqlen-1):
			x_embed = self.embeddings(preds[-1])
			x_hidden, (h_n, c_n) = self.lstm(x_embed.unsqueeze(dim=1), (h_n, c_n))
			x_probs = self.output_layer(x_hidden.squeeze(dim=1)).exp()
			preds.append(torch.multinomial(input=x_probs, num_samples=1).squeeze(dim=-1))
		return torch.stack(preds, dim=1), None


	def need_data_init(self):
		return False


	def info(self):
		return "LSTM with hidden size %i and %i layers" % (self.hidden_size, self.num_layers)

class TimeConcat(nn.Module):

	def __init__(self, time_embed, input_dp_rate=0.0):
		super().__init__()
		self.time_embed_layer = time_embed
		self.input_dropout = InputDropout(input_dp_rate)

	def forward(self, x, time_embed=None, length_one_hot=None, length=None):
		if time_embed is None:
			if length_one_hot is None and length is not None:
				length_one_hot = categorical_util.create_T_one_hot(length, dataset_max_len=int(self.time_embed_layer.weight.data.shape[1]//2))
			time_embed = self.time_embed_layer(length_one_hot)
		x = self.input_dropout(x)
		return torch.cat([x, time_embed], dim=-1)

class InputDropout(nn.Module):
	"""
	Removes input vectors with a probability of inp_dp_rate
	"""

	def __init__(self, dp_rate=0.0, scale_features=False):
		super().__init__()
		self.dp_rate = dp_rate
		self.scale_features = scale_features

	def forward(self, x):
		if not self.training:
			return x
		else:
			dp_mask = x.new_zeros(x.size(0), x.size(1), 1)
			dp_mask.bernoulli_(p=self.dp_rate)
			x = x * (1 - dp_mask)

			if self.scale_features:
				x = x * 1.0 / (1.0 - self.dp_rate)
			return x


## categorical flow model
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
		return self.forward(z, reverse)


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
		print("%s with %i flow layers" % (self.name, len(self.flow_layers)))
		print("-"*num_tokens)
		print("\n".join(layer_descp))
		print("="*num_tokens)

class FlowLanguageModeling(FlowModel):

	def __init__(self, model_params, dataset_class, vocab_size, vocab):
		super().__init__(layers=None, name="Categorical Language Modeling Flow")
		self.model_params = model_params
		self.dataset_class = dataset_class
		self.max_seq_len = self.model_params["max_seq_len"]
		self.vocab_size = vocab_size
		self.vocab = vocab

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
		self.encoding_layer = create_encoding(self.model_params["categ_encoding"], 
											  dataset_class=self.dataset_class, 
											  vocab_size=self.vocab_size,
											  vocab=self.vocab)

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
			kwargs["src_key_padding_mask"] = categorical_util.create_transformer_mask(length)
			kwargs["channel_padding_mask"] = categorical_util.create_channel_mask(length)
		return super().forward(z, ldj=ldj, reverse=reverse, length=length, **kwargs)


	def initialize_data_dependent(self, batch_list):
		# Batch list needs to consist of tuples: (z, kwargs)
		print("Initializing data dependent...")
		with torch.no_grad():
			for batch, kwargs in batch_list:
				kwargs["src_key_padding_mask"] = categorical_util.create_transformer_mask(kwargs["length"])
				kwargs["channel_padding_mask"] = categorical_util.create_channel_mask(kwargs["length"])
			for layer_index, layer in enumerate(self.flow_layers):
				batch_list = FlowModel.run_data_init_layer(batch_list, layer)
				# print(batch_list[0][0].shape)

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

class LSTMFeatureModel(nn.Module):

	def __init__(self, c_in, c_out, hidden_size, max_seq_len,
					   num_layers=1, dp_rate=0.0, 
					   input_dp_rate=0.0, **kwargs):
		super().__init__()

		time_embed = nn.Linear(2*max_seq_len, int(hidden_size//8))
		time_embed_dim = time_embed.weight.data.shape[0]
		self.time_concat = TimeConcat(time_embed=time_embed, 
									  input_dp_rate=input_dp_rate)
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

		lstm_out = categorical_util.run_padded_LSTM(x=embed, lstm_cell=self.lstm_module, length=length)

		out = self.net(lstm_out)
		out = self.out_layer(features=out, _inps=x)
		if channel_padding_mask is not None:
			out = out * channel_padding_mask
		return out

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

## flow layer
class FlowLayer(nn.Module):

	def __init__(self):
		super().__init__()


	def forward(self, z, ldj=None, reverse=False, **kwargs):
		raise NotImplementedError


	def reverse(self, z, ldj=None, **kwargs):
		return self.forward(z, ldj, reverse=True, **kwargs)


	def need_data_init(self):
		# This function indicates whether a specific flow needs a data-dependent initialization
		# or not. For instance, activation normalization requires such a initialization
		return False


	def data_init_forward(self, input_data, **kwargs):
		# Only necessary if need_data_init is True. Contains processing of data initialization
		raise NotImplementedError


	def info(self):
		# Function to retrieve small summary/info string
		raise NotImplementedError

class InvertibleConv(FlowLayer):

	def __init__(self, c_in, LU_decomposed=True):
		super().__init__()
		self.num_channels = c_in
		self.LU_decomposed = LU_decomposed

		if c_in == 2:
			# If we want a invertible convolution for 2 channels, there is a reasonable chance that a random
			# orthogonal matrix ends up similar to the identity matrix. To prevent such cases, we initialize
			# the weight matrix with a rotation matrix with an angle of 45-135 or 225-315 degrees
			rand_uni = np.random.uniform()
			if rand_uni < 0.5: 
				rand_uni = rand_uni * 2
				angle = 0.25 * np.pi + 0.5 * np.pi * rand_uni
			else:
				rand_uni = rand_uni * 2 - 1
				angle = 1.25 * np.pi + 0.5 * np.pi * rand_uni
			w_init = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
		else:
			# Initialize with a random orthogonal matrix
			w_init = np.random.randn(self.num_channels, self.num_channels)
			w_init = np.linalg.qr(w_init)[0].astype(np.float32)

		if not self.LU_decomposed:
			self.weight = nn.Parameter(torch.from_numpy(w_init), requires_grad=True)
		else: 
			# LU decomposition can slightly speed up the inverse
			np_p, np_l, np_u = scipy.linalg.lu(w_init)
			np_s = np.diag(np_u)
			np_sign_s = np.sign(np_s)
			np_log_s = np.log(np.abs(np_s))
			np_u = np.triu(np_u, k=1)
			l_mask = np.tril(np.ones(w_init.shape, dtype=np.float32), -1)
			eye = np.eye(*w_init.shape, dtype=np.float32)

			self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
			self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
			self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)), requires_grad=True)
			self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)), requires_grad=True)
			self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)), requires_grad=True)
			self.register_buffer('l_mask', torch.Tensor(l_mask))
			self.register_buffer('eye', torch.Tensor(eye))

		self.eval_dict = defaultdict(lambda : self._get_default_inner_dict())

	def _get_default_inner_dict(self):
		return {"weight": None, "inv_weight": None, "sldj": None}

	def _get_weight(self, device_name, inverse=False):
		if self.training or self._is_eval_dict_empty(device_name):
			if not self.LU_decomposed:
				weight = self.weight
				sldj = torch.slogdet(weight)[1]
			else:
				l, log_s, u = self.l, self.log_s, self.u
				l = l * self.l_mask + self.eye
				u = u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(log_s))
				weight = torch.matmul(self.p, torch.matmul(l, u))
				sldj = log_s.sum()
		
		if not self.training:
			if self._is_eval_dict_empty(device_name):
				self.eval_dict[device_name]["weight"] = weight.detach()
				self.eval_dict[device_name]["sldj"] = sldj.detach()
				self.eval_dict[device_name]["inv_weight"] = torch.inverse(weight.double()).float().detach()
			else:
				weight, sldj = self.eval_dict[device_name]["weight"], self.eval_dict[device_name]["sldj"]
		elif not self._is_eval_dict_empty(device_name):
			self._empty_eval_dict(device_name)
		
		if inverse:
			if self.training:
				weight = torch.inverse(weight.double()).float()
			else:
				weight = self.eval_dict[device_name]["inv_weight"]
		
		return weight, sldj


	def _is_eval_dict_empty(self, device_name=None):
		if device_name is not None:
			return not (device_name in self.eval_dict)
		else:
			return len(self.eval_dict) == 0


	def _empty_eval_dict(self, device_name=None):
		if device_name is not None:
			self.eval_dict.pop(device_name)
		else:
			self.eval_dict = defaultdict(lambda : self._get_default_inner_dict())
		

	def forward(self, x, ldj=None, reverse=False, length=None, channel_padding_mask=None, layer_share_dict=None, **kwargs):
		if ldj is None:
			ldj = x.new_zeros(x.size(0),) 
		length = x.size(1) if length is None else length.float()

		weight, sldj = self._get_weight(device_name=str(x.device), inverse=reverse)
		sldj = sldj * length # No batch because LDJ is vector over batch

		if reverse:
			ldj = ldj - sldj
		else:
			ldj = ldj + sldj

		z = torch.matmul(x, weight.unsqueeze(dim=0))
		if channel_padding_mask is not None:
			z = z * channel_padding_mask

		if layer_share_dict is not None:
			layer_share_dict["t"] = layer_share_dict["t"] * 0.0
			layer_share_dict["log_s"] = layer_share_dict["log_s"] * 0.0
			if "error_decay" in layer_share_dict:
				layer_share_dict["error_decay"] = layer_share_dict["error_decay"] * 0.0

		assert torch.isnan(ldj).sum() == 0 and torch.isnan(z).sum() == 0, \
			   "[!] ERROR: Output of invertible 1x1 convolution contains NaN.\n" + \
			   "-> LDJ: %s\n" % str(torch.isnan(ldj).sum().item()) + \
			   "-> SLDJ: %s\n" % str(torch.isnan(sldj).sum().item()) + \
			   "-> Z: %s\n" % str(torch.isnan(z).sum().item()) + \
			   "-> Weight:" + str(weight.detach().cpu().numpy())
		
		return z, ldj


	def info(self):
		return "Invertible 1x1 Convolution - %i channels %s" % (self.num_channels, "(LU decomposed)" if self.LU_decomposed else "")

class ActNormFlow(FlowLayer):
	"""
	Normalizes the activations over channels
	"""

	def __init__(self, c_in, data_init=True):
		super().__init__()
		self.c_in = c_in 
		self.data_init = data_init

		self.bias = nn.Parameter(torch.zeros(1, 1, self.c_in))
		self.scales = nn.Parameter(torch.zeros(1, 1, self.c_in))


	def forward(self, z, ldj=None, reverse=False, length=None, channel_padding_mask=None, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		if length is None:
			if channel_padding_mask is None:
				length = z.size(1)
			else:
				length = channel_padding_mask.squeeze(dim=2).sum(dim=1)
		else:
			length = length.float()
		
		if not reverse:
			z = (z + self.bias) * torch.exp(self.scales)
			ldj += self.scales.sum(dim=[1,2]) * length
		else:
			z = z * torch.exp(-self.scales) - self.bias
			ldj += (-self.scales.sum(dim=[1,2])) * length

		if channel_padding_mask is not None:
			z = z * channel_padding_mask

		assert torch.isnan(z).sum() == 0, "[!] ERROR: z contains NaN values."
		assert torch.isnan(ldj).sum() == 0, "[!] ERROR: ldj contains NaN values."

		return z, ldj


	def need_data_init(self):
		return self.data_init


	def data_init_forward(self, input_data, channel_padding_mask=None, **kwargs):
		if channel_padding_mask is None:
			channel_padding_mask = input_data.new_ones(input_data.shape)
		mask = channel_padding_mask
		num_exp = mask.sum(dim=[0,1], keepdims=True)
		masked_input = input_data * mask

		bias_init = -masked_input.sum(dim=[0,1], keepdims=True) / num_exp
		self.bias.data = bias_init

		var_data = ( ( (input_data + bias_init)**2 ) * mask).sum(dim=[0,1], keepdims=True) / num_exp
		scaling_init = -0.5*var_data.log()
		self.scales.data = scaling_init

		out = (masked_input + self.bias) * torch.exp(self.scales)
		out_mean = (out*mask).sum(dim=[0,1]) / num_exp.squeeze()
		out_var = torch.sqrt(( ( (out - out_mean)**2 ) * mask).sum(dim=[0,1]) / num_exp)
		print("[INFO - ActNorm] New mean", out_mean)
		print("[INFO - ActNorm] New variance", out_var)

	def info(self):
		return "Activation Normalizing Flow (c_in=%i)" % (self.c_in)

class ExtActNormFlow(FlowLayer):
	"""
	Normalizes the activations over channels
	"""

	def __init__(self, c_in, net, zero_init=False, data_init=False, make_unique=False):
		super().__init__()
		self.c_in = c_in 
		self.data_init = data_init
		self.make_unique = make_unique

		self.pred_net = net
		if zero_init:
			if hasattr(self.pred_net, "initialize_zeros"):
				self.pred_net.initialize_zeros()
			elif isinstance(self.pred_net, nn.Sequential):
				self.pred_net[-1].weight.data.zero_()
				self.pred_net[-1].bias.data.zero_()


	def _run_nn(self, ext_input):
		if not self.make_unique:
			return self.pred_net(ext_input)
		else:
			orig_shape = ext_input.shape
			unique_inputs = torch.unique(ext_input)
			unique_outs = self.pred_net(unique_inputs)
			unique_inputs = unique_inputs.view(1, -1)
			ext_input = ext_input.view(-1, 1)
			indices = ((ext_input == unique_inputs).long() * torch.arange(unique_inputs.shape[1], dtype=torch.long, device=ext_input.device).unsqueeze(dim=0)).sum(dim=1)
			ext_out = unique_outs.index_select(dim=0, index=indices)
			ext_out = ext_out.reshape(orig_shape + unique_outs.shape[-1:])
			return ext_out


	def forward(self, z, ldj=None, reverse=False, ext_input=None, channel_padding_mask=None, layer_share_dict=None, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		if channel_padding_mask is None:
			channel_padding_mask = 1.0

		if ext_input is None:
			print("[!] WARNING: External input in ExtActNormFlow is None. Using default params...")
			bias = z.new_zeros(z.size(0), z.size(1), z.size(2))
			scales = bias
		else:
			nn_out = self._run_nn(ext_input)
			bias, scales = nn_out.chunk(2, dim=2)
			scales = torch.tanh(scales)
		
		if not reverse:
			z = (z + bias) * torch.exp(scales)
			ldj += (scales * channel_padding_mask).sum(dim=[1,2]) 
			if layer_share_dict is not None:
				layer_share_dict["t"] = (layer_share_dict["t"] + bias) * torch.exp(scales)
				layer_share_dict["log_s"] = layer_share_dict["log_s"] + scales
		else:
			z = z * torch.exp(-scales) - bias
			ldj += -(scales * channel_padding_mask).sum(dim=[1,2])

		assert torch.isnan(z).sum() == 0, "[!] ERROR: z contains NaN values."
		assert torch.isnan(ldj).sum() == 0, "[!] ERROR: ldj contains NaN values."
	
		return z, ldj


	def need_data_init(self):
		return self.data_init


	def data_init_forward(self, input_data, channel_padding_mask=None, **kwargs):
		if channel_padding_mask is None:
			channel_padding_mask = input_data.new_ones(input_data.shape)
		else:
			channel_padding_mask = channel_padding_mask.view(input_data.shape[:-1] + channel_padding_mask.shape[-1:])
		mask = channel_padding_mask
		num_exp = mask.sum(dim=[0,1], keepdims=True)
		masked_input = input_data

		bias_init = -masked_input.sum(dim=[0,1], keepdims=True) / num_exp

		var_data = ( ( (input_data + bias_init)**2 ) * mask).sum(dim=[0,1], keepdims=True) / num_exp
		scaling_init = -0.5*var_data.log()

		bias = torch.cat([bias_init, scaling_init], dim=-1).squeeze()

		if isinstance(self.pred_net, nn.Sequential):
			self.pred_net[-1].bias.data = bias
		else:
			self.pred_net.set_bias(bias)

		out = (masked_input + bias_init) * torch.exp(scaling_init)
		out_mean = (out*mask).sum(dim=[0,1]) / num_exp.squeeze()
		out_var = torch.sqrt(( ( (out - out_mean)**2 ) * mask).sum(dim=[0,1]) / num_exp)
		print("[INFO - External ActNorm] New mean", out_mean)
		print("[INFO - External ActNorm] New variance", out_var)


	def info(self):
		return "External Activation Normalizing Flow (c_in=%i)" % (self.c_in)

class CouplingLayer(FlowLayer):

	def __init__(self, c_in, mask, 
					   model_func,
					   block_type=None,
					   c_out=-1,
					   **kwargs):
		super().__init__()
		self.c_in = c_in
		self.c_out = c_out if c_out > 0 else 2 * c_in
		self.register_buffer('mask', mask)
		self.block_type = block_type

		# Scaling factor
		self.scaling_factor = nn.Parameter(torch.zeros(c_in))
		self.nn = model_func(c_out=self.c_out)

	def run_network(self, x, length=None, **kwargs):
		if isinstance(self.nn, nn.Sequential):
			nn_out = categorical_util.run_sequential_with_mask(self.nn, x, 
											  length=length, 
											  **kwargs)
		else:
			nn_out = self.nn(x, length=length,
							 **kwargs)

		if "channel_padding_mask" in kwargs and kwargs["channel_padding_mask"] is not None:
			nn_out = nn_out * kwargs["channel_padding_mask"]
		return nn_out


	def forward(self, z, ldj=None, reverse=False, channel_padding_mask=None, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		if channel_padding_mask is None:
			channel_padding_mask = torch.ones_like(z)

		mask = self._prepare_mask(self.mask, z)
		z_in = z * mask

		nn_out = self.run_network(x=z_in, **kwargs)

		nn_out = nn_out.view(nn_out.shape[:-1] + (nn_out.shape[-1]//2, 2))
		s, t = nn_out[...,0], nn_out[...,1]
		
		scaling_fac = self.scaling_factor.exp().view([1, 1, s.size(-1)])
		s = torch.tanh(s / scaling_fac.clamp(min=1.0)) * scaling_fac
		
		s = s * (1 - mask)
		t = t * (1 - mask)

		z, layer_ldj = CouplingLayer.run_with_params(z, s, t, reverse=reverse)
		ldj = ldj + layer_ldj

		return z, ldj # , detail_dict

	def _prepare_mask(self, mask, z):
		# Mask input so that we only use the un-masked regions as input
		mask = self.mask.unsqueeze(dim=0) if len(z.shape) > len(self.mask.shape) else self.mask
		if mask.size(1) < z.size(1) and mask.size(1) > 1:
			mask = mask.repeat(1, int(math.ceil(z.size(1)/mask.size(1))), 1).contiguous()
		if mask.size(1) > z.size(1):
			mask = mask[:,:z.size(1)]
		return mask

	@staticmethod
	def get_coup_params(nn_out, mask, scaling_factor=None):
		nn_out = nn_out.view(nn_out.shape[:-1] + (nn_out.shape[-1]//2, 2))
		s, t = nn_out[...,0], nn_out[...,1]
		if scaling_factor is not None:
			scaling_fac = scaling_factor.exp().view([1, 1, s.size(-1)])
			s = torch.tanh(s / scaling_fac.clamp(min=1.0)) * scaling_fac
		
		s = s * (1 - mask)
		t = t * (1 - mask)
		return s, t

	@staticmethod
	def run_with_params(orig_z, s, t, reverse=False):
		if not reverse:
			scale = torch.exp(s)
			z_out = (orig_z + t) * scale
			ldj = s.sum(dim=[1,2])
		else:
			inv_scale = torch.exp(-1 * s)
			z_out = orig_z * inv_scale - t
			ldj = -s.sum(dim=[1,2])
		return z_out, ldj


	@staticmethod
	def create_channel_mask(c_in, ratio=0.5, mask_floor=True):
		"""
		Ratio: number of channels that are alternated/for which we predict parameters
		"""
		if mask_floor:
			c_masked = int(math.floor(c_in * ratio))
		else:
			c_masked = int(math.ceil(c_in * ratio))
		c_unmasked = c_in - c_masked
		mask = torch.cat([torch.ones(1, c_masked), torch.zeros(1, c_unmasked)], dim=1)
		return mask


	@staticmethod
	def create_chess_mask(seq_len=2):
		assert seq_len > 1
		seq_unmask = int(seq_len // 2)
		seq_mask = seq_len - seq_unmask
		mask = torch.cat([torch.ones(seq_mask, 1), torch.zeros(seq_unmask, 1)], dim=1).view(-1, 1)
		return mask


	def info(self):
		is_channel_mask = (self.mask.size(0) == 1)
		info_str = "Coupling Layer - Input size %i" % (self.c_in)
		if self.block_type is not None:
			info_str += ", block type %s" % (self.block_type)
		info_str += ", mask ratio %.2f, %s mask" % ((1-self.mask).mean().item(), "channel" if is_channel_mask else "chess")
		return info_str

class MixtureCDFCoupling(CouplingLayer):

	def __init__(self, c_in, mask, 
					   model_func, 
					   block_type=None, 
					   num_mixtures=10, 
					   regularizer_max=-1,
					   regularizer_factor=1, **kwargs):
		"""
		Logistic mixture coupling layer as applied in Flow++. 
		Parameters:
			c_in - Number of input channels
			mask - Mask to apply on the input. 1 means that the element is used as input, 0 that it is transformed
			model_func - Function for creating a model. Needs to take as input argument the number of output channels
			block_type - Name of the model. Only used for printing
			num_mixtures - Number of mixtures to apply in the layer
			regularizer_max - Mixture coupling layers apply a iterative algorithm to invert the transformations, which
							  is limited in precision. To prevent precision errors, we regularize the CDF to be between
							  10^(-regularizer_max) and 1-10^(-regularizer_max). A value of 3.5 usually works well without
							  any noticable decrease in performance. Default of -1 means no regularization.
							  This parameter should be used if sampling is important (e.g. in molecule generation)
			regularizer_factor - Factor with which to multiply the regularization loss. Commonly a value of 1 or 2 works well.
		"""
		super().__init__(c_in=c_in, mask=mask, 
						 model_func=model_func,
						 block_type=block_type,
						 c_out=c_in*(2 + num_mixtures * 3),
						 **kwargs)
		self.num_mixtures = num_mixtures
		self.mixture_scaling_factor = nn.Parameter(torch.zeros(self.c_in, self.num_mixtures))
		self.regularizer_max = regularizer_max
		self.regularizer_factor = regularizer_factor


	def forward(self, z, ldj=None, reverse=False, channel_padding_mask=None, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		if channel_padding_mask is None:
			channel_padding_mask = torch.ones_like(z)

		# Mask input so that we only use the un-masked regions as input
		orig_z = z
		mask = self._prepare_mask(self.mask, z)
		z_in = z * mask
		
		nn_out = self.run_network(x=z_in, **kwargs)

		t, log_s, log_pi, mixt_t, mixt_log_s = MixtureCDFCoupling.get_mixt_params(nn_out, mask, 
																				 num_mixtures=self.num_mixtures,
																				 scaling_factor=self.scaling_factor,
																				 mixture_scaling_factor=self.mixture_scaling_factor)
		orig_z = orig_z.double()
		z_out, ldj, reg_ldj = MixtureCDFCoupling.run_with_params(orig_z=orig_z,
																  t=t, log_s=log_s, log_pi=log_pi,
																  mixt_t=mixt_t, mixt_log_s=mixt_log_s,
																  reverse=reverse,
																  is_training=self.training,
																  reg_max=self.regularizer_max,
																  reg_factor=self.regularizer_factor,
																  mask=mask,
																  channel_padding_mask=channel_padding_mask,
																  return_reg_ldj=True)
			
		z_out = z_out.float()
		ldj = ldj.float()
		z_out = z_out * channel_padding_mask

		detail_out = {"ldj": ldj}
		if reg_ldj is not None:
			detail_out["regularizer_ldj"] = reg_ldj.float().sum(dim=[1,2])

		assert torch.isnan(z_out).sum() == 0 and torch.isnan(ldj).sum() == 0, "[!] ERROR: Found NaN in Mixture Coupling layer. Layer info: %s\n" % self.info() + \
				"LDJ NaN: %s, Z out NaN: %s, Z in NaN: %s, NN out NaN: %s\n" % (str(torch.isnan(ldj).sum().item()), str(torch.isnan(z_out).sum().item()), str(torch.isnan(orig_z).sum().item()), str(torch.isnan(nn_out).sum().item())) + \
				"Max/Min transition t: %s / %s\n" % (str(t.max().item()), str(t.min().item())) + \
				"Max/Min log scaling s: %s / %s\n" % (str(log_s.max().item()), str(log_s.min().item())) + \
				"Max/Min log pi: %s / %s\n" % (str(log_pi.max().item()), str(log_pi.min().item())) + \
				"Max/Min mixt t: %s / %s\n" % (str(mixt_t.max().item()), str(mixt_t.min().item())) + \
				"Max/Min mixt log s: %s / %s\n" % (str(mixt_log_s.max().item()), str(mixt_log_s.min().item())) + \
				"Mixt ldj NaN: %s\n" % (str(torch.isnan(mixt_ldj).sum().item())) + \
				"Logistic ldj NaN: %s\n" % (str(torch.isnan(logistic_ldj).sum().item()))

		return z_out, ldj, detail_out


	@staticmethod
	def run_with_params(orig_z, t, log_s, log_pi, mixt_t, mixt_log_s, reverse=False, 
						reg_max=-1, reg_factor=1, mask=None, channel_padding_mask=None,
						is_training=True, return_reg_ldj=False):
		change_mask = 1-mask if mask is not None else torch.ones_like(orig_z)
		if channel_padding_mask is not None:
			change_mask = change_mask * channel_padding_mask
		reg_ldj = None
		if not reverse:
			# Calculate CDF function for given mixtures and input
			z_out = categorical_util.mixture_log_cdf(x=orig_z, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s).exp()

			# Regularize mixtures if wanted (only done during training as LDJ is increased)
			if reg_max > 0 and is_training:
				reg_ldj = torch.stack([safe_log(z_out), safe_log(1-z_out)], dim=-1)/np.log(10) # Change to 10 base
				reg_ldj = reg_ldj.clamp(max=-reg_max) + reg_max
				reg_ldj = reg_ldj.sum(dim=-1)
				reg_ldj = reg_ldj * change_mask
			else:
				reg_ldj = torch.zeros_like(z_out)

			# Map from [0,1] domain back to [-inf,inf] by inverse sigmoid
			z_out, mixt_ldj = categorical_util.inverse(z_out)
			# Output affine transformation
			z_out = (z_out + t) * log_s.exp()
			# Determine LDJ of the transformation
			logistic_ldj = categorical_util.mixture_log_pdf(orig_z, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s)
			# Combine all LDJs
			ldj = (change_mask * (log_s + mixt_ldj + logistic_ldj + reg_ldj * reg_factor)).sum(dim=[1,2])
		else:
			# Reverse output affine transformation
			z_out = orig_z * (-log_s).exp() - t
			# Apply sigmoid to map back to [0,1] domain
			z_out, mixt_ldj = categorical_util.inverse(z_out, reverse=True)
			# Clamping to prevent numerical instabilities
			z_out = z_out.clamp(1e-5, 1. - 1e-5)
			# Inverse the cummulative distribution function of mixtures. Iterative algorithm, maps back to [-inf,inf]
			z_out = categorical_util.mixture_inv_cdf(z_out, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s)
			# Determine LDJ of this transformation
			logistic_ldj = categorical_util.mixture_log_pdf(z_out, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s)
			# Combine all LDJs (note the negative sign)
			ldj = -(change_mask * (log_s + mixt_ldj + logistic_ldj)).sum(dim=[1,2])
		if mask is not None: # Applied to ensure that the masked elements are not changed by numerical inaccuracies
			z_out = z_out * change_mask + orig_z * (1 - change_mask)
		if return_reg_ldj:
			return z_out, ldj, reg_ldj
		else:
			return z_out, ldj


	@staticmethod
	def get_mixt_params(nn_out, mask, num_mixtures, scaling_factor=None, mixture_scaling_factor=None):
		# Split network output into transformation parameters
		param_num = 2 + num_mixtures * 3
		nn_out = nn_out.reshape(nn_out.shape[:-1] + (nn_out.shape[-1]//param_num, param_num))
		t = nn_out[..., 0]
		log_s = nn_out[..., 1]
		log_pi = nn_out[..., 2:2+num_mixtures]
		mixt_t = nn_out[..., 2+num_mixtures:2+2*num_mixtures]
		mixt_log_s = nn_out[..., 2+2*num_mixtures:2+3*num_mixtures]

		# Stabilizing the scaling
		if scaling_factor is not None:
			scaling_fac = scaling_factor.exp().view(*(tuple([1 for _ in range(len(log_s.shape)-1)])+scaling_factor.shape))
			log_s = torch.tanh(log_s / scaling_fac.clamp(min=1.0)) * scaling_fac
		if mixture_scaling_factor is not None:
			mixt_fac = mixture_scaling_factor.exp().view(*(tuple([1 for _ in range(len(mixt_log_s.shape)-2)])+mixture_scaling_factor.shape))
			mixt_log_s = torch.tanh(mixt_log_s / mixt_fac.clamp(min=1.0)) * mixt_fac

		# Masking parameters
		if mask is not None:
			t = t * (1 - mask)
			log_s = log_s * (1 - mask)
			mask_ext = mask.unsqueeze(dim=-1)
			log_pi = log_pi * (1 - mask_ext) # Not strictly necessary but done for safety
			mixt_t = mixt_t * (1 - mask_ext)
			mixt_log_s = mixt_log_s * (1 - mask_ext)

		# Converting to double to prevent any numerical issues
		t = t.double()
		log_s = log_s.double()
		log_pi = log_pi.double()
		mixt_t = mixt_t.double()
		mixt_log_s = mixt_log_s.double()

		return t, log_s, log_pi, mixt_t, mixt_log_s


	def info(self):
		is_channel_mask = (self.mask.size(0) == 1)
		info_str = "Mixture CDF Coupling Layer - Input size %i" % (self.c_in)
		if self.block_type is not None:
			info_str += ", block type %s" % (self.block_type)
		info_str += ", %i mixtures" % (self.num_mixtures) + \
					", mask ratio %.2f, %s mask" % ((1-self.mask).mean().item(), "channel" if is_channel_mask else "chess")
		return info_str

class AutoregressiveMixtureCDFCoupling(FlowLayer):

	def __init__(self, c_in, model_func, block_type=None, num_mixtures=10):
		super().__init__()
		self.c_in = c_in
		self.num_mixtures = num_mixtures
		self.block_type = block_type
		self.scaling_factor = nn.Parameter(torch.zeros(self.c_in))
		self.mixture_scaling_factor = nn.Parameter(torch.zeros(self.c_in, self.num_mixtures))
		self.nn = model_func(c_out=c_in*(2 + 3 * self.num_mixtures))


	def forward(self, z, ldj=None, reverse=False, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		
		if not reverse:
			nn_out = self.nn(x=z, **kwargs)

			t, log_s, log_pi, mixt_t, mixt_log_s = MixtureCDFCoupling.get_mixt_params(nn_out, mask=None,
																				 num_mixtures=self.num_mixtures,
																				 scaling_factor=self.scaling_factor,
																				 mixture_scaling_factor=self.mixture_scaling_factor)
			
			z = z.double()
			z_out, ldj_mixt = MixtureCDFCoupling.run_with_params(z, t, log_s, log_pi, mixt_t, mixt_log_s, reverse=reverse)
		else:
			raise NotImplementedError

		ldj = ldj + ldj_mixt.float()	
		z_out = z_out.float()
		if "channel_padding_mask" in kwargs and kwargs["channel_padding_mask"] is not None:
			z_out = z_out * kwargs["channel_padding_mask"]

		return z_out, ldj


	def info(self):
		s = "Autoregressive Mixture CDF Coupling Layer - Input size %i" % (self.c_in)
		if self.block_type is not None:
			s += ", block type %s" % (self.block_type)
		return s


## create_encoding
def create_encoding(encoding_params, dataset_class, vocab=None, vocab_size=-1, category_prior=None):
	assert not (vocab is None and vocab_size <= 0), "[!] ERROR: When creating the encoding, either a torchtext vocabulary or the vocabulary size needs to be passed."
	use_dequantization = encoding_params.pop("use_dequantization")
	use_variational = encoding_params.pop("use_variational")

	if use_dequantization and "model_func" not in encoding_params["flow_config"]:
		print("[#] WARNING: For using variational dequantization as encoding scheme, a model function needs to be specified" + \
			  " in the encoding parameters, key \"flow_config\" which was missing here. Will deactivate dequantization...")
		use_dequantization = False

	if use_dequantization:
		encoding_flow = VariationalDequantization
	elif use_variational:
		encoding_flow = VariationalCategoricalEncoding
	else:
		encoding_flow = LinearCategoricalEncoding

	return encoding_flow(dataset_class=dataset_class,
						 vocab=vocab,
						 vocab_size=vocab_size,
						 category_prior=category_prior,
						 **encoding_params)

class LinearCategoricalEncoding(FlowLayer):
	"""
	Class for implementing the mixture model and linear flow encoding scheme of Categorical Normalizing Flows.
	A mixture model can be achieved by using a single activation normalization layer as "linear flow".
	Hence, this class combines both encoding schemes. 
	"""

	def __init__(self, num_dimensions, flow_config,
					   dataset_class=None,
					   vocab=None, vocab_size=-1, 
					   use_decoder=False, decoder_config=None, 
					   default_embed_layer_dims=64,
					   category_prior=None,
					   **kwargs):
		super().__init__()
		self.use_decoder = use_decoder
		self.dataset_class = dataset_class
		self.D = num_dimensions

		self.embed_layer, self.vocab_size = categorical_util.create_embed_layer(vocab, vocab_size, default_embed_layer_dims)
		self.num_categories = self.vocab_size
	
		self.prior_distribution = categorical_util.LogisticDistribution(mu=0.0, sigma=1.0) # Prior distribution in encoding flows
		self.flow_layers = self._create_flows(num_dims=num_dimensions, 
										 embed_dims=self.embed_layer.weight.shape[1], 
										 config=flow_config)
		# Create decoder if needed
		if self.use_decoder:
			self.decoder = self.create_decoder(num_categories=self.vocab_size, 
										   num_dims=self.D,
										   config=decoder_config)

		# Prior over the categories. If not given, a uniform prior is assumed
		if category_prior is None:
			category_prior = torch.zeros(self.vocab_size, dtype=torch.float32)
		else:
			assert category_prior.shape[0] == self.num_categories, "[!] ERROR: Category prior needs to be of size [%i] but is %s" % (self.num_categories, str(category_prior.shape))
			if isinstance(category_prior, np.ndarray):
				category_prior = torch.from_numpy(category_prior)
		self.register_buffer("category_prior", F.log_softmax(category_prior, dim=-1))

	def create_decoder(self, num_categories, num_dims, config, **kwargs):
		num_layers = categorical_util.get_param_val(config, "num_layers", 1)
		hidden_size = categorical_util.get_param_val(config, "hidden_size", 64)

		return DecoderLinear(num_categories, 
						 embed_dim=num_dims, 
						 hidden_size=hidden_size, 
						 num_layers=num_layers,
						 **kwargs)

	def _create_flows(self, num_dims, embed_dims, config):
		num_flows = categorical_util.get_param_val(config, "num_flows", 0)
		num_hidden_layers = categorical_util.get_param_val(config, "hidden_layers", 2)
		hidden_size = categorical_util.get_param_val(config, "hidden_size", 256)
		
		# We apply a linear net in the coupling layers for linear flows
		block_type_name = "LinearNet"
		block_fun_coup = lambda c_out : LinearNet(c_in=num_dims,
												c_out=c_out,
												num_layers=num_hidden_layers,
												hidden_size=hidden_size,
												ext_input_dims=embed_dims)

		# For the activation normalization, we map an embedding to scaling and bias with a single layer
		block_fun_actn = lambda : SimpleLinearLayer(c_in=embed_dims, c_out=2*num_dims, data_init=True)
		
		permut_layer = lambda flow_index : InvertibleConv(c_in=num_dims)
		actnorm_layer = lambda flow_index : ExtActNormFlow(c_in=num_dims, 
														net=block_fun_actn())
		# We do not use mixture coupling layers here aas we need the inverse to be differentiable as well
		coupling_layer = lambda flow_index : CouplingLayer(c_in=num_dims, 
														mask=CouplingLayer.create_channel_mask(c_in=num_dims), 
														block_type=block_type_name,
														model_func=block_fun_coup)

		flow_layers = []
		if num_flows == 0 or num_dims == 1: # Num_flows == 0 => mixture model, num_dims == 1 => coupling layers have no effect
			flow_layers += [actnorm_layer(flow_index=0)]
		else:
			for flow_index in range(num_flows):
				flow_layers += [
					actnorm_layer(flow_index), 
					permut_layer(flow_index),
					coupling_layer(flow_index)
				]

		return nn.ModuleList(flow_layers)

		
	def forward(self, z, ldj=None, reverse=False, beta=1, delta=0.0, channel_padding_mask=None, **kwargs):
		## We reshape z into [batch, 1, ...] as every categorical variable is considered to be independent.
		batch_size, seq_length = z.size(0), z.size(1)
		# print(z.shape) # [batch_size, seq_length]
		z = z.reshape((batch_size * seq_length, 1) + z.shape[2:])
		# print(z.shape) # [batch_size*seq_length, 1]
		if channel_padding_mask is not None:
			# print(channel_padding_mask.shape) # [batch_size, seq_length, 1]
			channel_padding_mask = channel_padding_mask.reshape(batch_size * seq_length, 1, -1)
			# print(channel_padding_mask.shape)
		else:
			channel_padding_mask = z.new_ones((batch_size * seq_length, 1, 1), dtype=torch.float32)

		ldj_loc = z.new_zeros(z.size(0), dtype=torch.float32)
		detailed_ldj = {}
		
		if not reverse:
			# z is of shape [Batch, SeqLength]
			z_categ = z # Renaming here for better readability (what is discrete and what is continuous)

			## 1.) Forward pass of current token flow
			z_cont = self.prior_distribution.sample(shape=(batch_size * seq_length, 1, self.D)).to(z_categ.device)
			init_log_p = self.prior_distribution.log_prob(z_cont).sum(dim=[1,2])
			z_cont, ldj_forward = self._flow_forward(z_cont, z_categ, reverse=False)

			## 2.) Approach-specific calculation of the posterior
			if not self.use_decoder:
				class_prior_log = torch.take(self.category_prior, z_categ.squeeze(dim=-1))
				log_point_prob = init_log_p - ldj_forward + class_prior_log
				class_prob_log = self._calculate_true_posterior(z_cont, z_categ, log_point_prob)
			else:
				class_prob_log = self._decoder_forward(z_cont, z_categ)

			## 3.) Calculate final LDJ
			ldj_loc = (beta * class_prob_log - (init_log_p - ldj_forward))
			ldj_loc = ldj_loc * channel_padding_mask.squeeze()
			z_cont = z_cont * channel_padding_mask
			z_out = z_cont

			# print(self.training)
			## 4.) Statistics for debugging/monotoring
			if self.training:
				with torch.no_grad():
					z_min = z_out.min()
					z_max = z_out.max()
					z_std = z_out.view(-1, z_out.shape[-1]).std(0).mean()
					channel_padding_mask = channel_padding_mask.squeeze()
					detailed_ldj = {"avg_token_prob": (class_prob_log.exp() * channel_padding_mask).sum()/channel_padding_mask.sum(), 
									"avg_token_bpd": -(class_prob_log * channel_padding_mask).sum()/channel_padding_mask.sum() * np.log2(np.exp(1)),
									"z_min": z_min,
									"z_max": z_max,
									"z_std": z_std}
					detailed_ldj = {key: val.detach() for key, val in detailed_ldj.items()}

		else:
			# z is of shape [Batch * seq_len, 1, D]
			assert z.size(-1) == self.D, "[!] ERROR in categorical decoding: Input must have %i latent dimensions but got %i" % (self.D, z.shape[-1])

			class_prior_log = self.category_prior[None,None,:]
			z_cont = z

			if not self.use_decoder:
				z_out = self._posterior_sample(z_cont)
			else:
				z_out = self._decoder_sample(z_cont)

		# Reshape output back to original shape
		if not reverse:
			z_out = z_out.reshape(batch_size, seq_length, -1)
		else:
			z_out = z_out.reshape(batch_size, seq_length)
		ldj_loc = ldj_loc.reshape(batch_size, seq_length).sum(dim=-1)
		
		# Add LDJ 
		if ldj is not None:
			ldj = ldj + ldj_loc
		else:
			ldj = ldj_loc

		return z_out, ldj, detailed_ldj


	def _flow_forward(self, z_cont, z_categ, reverse, **kwargs):
		ldj = z_cont.new_zeros(z_cont.size(0), dtype=torch.float32)
		embed_features = self.embed_layer(z_categ)
		# print(embed_features.shape)
		
		for flow in (self.flow_layers if not reverse else reversed(self.flow_layers)):
			z_cont, ldj = flow(z_cont, ldj, ext_input=embed_features, reverse=reverse, **kwargs)

		return z_cont, ldj


	def _decoder_forward(self, z_cont, z_categ, **kwargs):
		## Applies the deocder on every continuous variable independently and return probability of GT class
		class_prob_log = self.decoder(z_cont)
		class_prob_log = class_prob_log.gather(dim=-1, index=z_categ.view(-1,1))
		return class_prob_log


	def _calculate_true_posterior(self, z_cont, z_categ, log_point_prob, **kwargs):
		## Run backward pass of *all* class-conditional flows
		z_back_in = z_cont.expand(-1, self.num_categories, -1).reshape(-1, 1, z_cont.size(2))
		sample_categ = torch.arange(self.num_categories, dtype=torch.long).to(z_cont.device)
		sample_categ = sample_categ[None,:].expand(z_categ.size(0), -1).reshape(-1, 1)

		z_back, ldj_backward = self._flow_forward(z_back_in, sample_categ, reverse=True, **kwargs)
		back_log_p = self.prior_distribution.log_prob(z_back).sum(dim=[1,2])
		
		## Calculate the denominator (sum of probabilities of all classes)
		flow_log_prob = back_log_p + ldj_backward
		log_prob_denominator = flow_log_prob.view(z_cont.size(0), self.num_categories) + self.category_prior[None,:]
		# Replace log_prob of original class with forward probability
		# This improves stability and prevents the model to exploit numerical errors during inverting the flows
		orig_class_mask = categorical_util.one_hot(z_categ.squeeze(), num_classes=log_prob_denominator.size(1))
		log_prob_denominator = log_prob_denominator * (1 - orig_class_mask) + log_point_prob.unsqueeze(dim=-1) * orig_class_mask
		# Denominator is the sum of probability -> turn log to exp, and back to log
		log_denominator = torch.logsumexp(log_prob_denominator, dim=-1)
		
		## Combine nominator and denominator for final prob log
		class_prob_log = (log_point_prob - log_denominator)
		return class_prob_log


	def _decoder_sample(self, z_cont, **kwargs):
		## Sampling from decoder by taking the argmax.
		# We could also sample from the probabilities, however experienced that the argmax gives more stable results.
		# Presumably because the decoder has also seen values sampled from the encoding distributions and not anywhere besides that.
		return self.decoder(z_cont).argmax(dim=-1)


	def _posterior_sample(self, z_cont, **kwargs):
		## Run backward pass of *all* class-conditional flows
		z_back_in = z_cont.expand(-1, self.num_categories, -1).reshape(-1, 1, z_cont.size(2))
		sample_categ = torch.arange(self.num_categories, dtype=torch.long).to(z_cont.device)
		sample_categ = sample_categ[None,:].expand(z_cont.size(0), -1).reshape(-1, 1)

		z_back, ldj_backward = self._flow_forward(z_back_in, sample_categ, reverse=True, **kwargs)
		back_log_p = self.prior_distribution.log_prob(z_back).sum(dim=[1,2])
		
		## Calculate the log probability for each class
		flow_log_prob = back_log_p + ldj_backward
		log_prob_denominator = flow_log_prob.view(z_cont.size(0), self.num_categories) + self.category_prior[None,:]
		return log_prob_denominator.argmax(dim=-1)


	def info(self):
		s = ""
		if len(self.flow_layers) > 1:
			s += "Linear Encodings of categories, with %i dimensions and %i flows.\n" % (self.D, len(self.flow_layers))
		else:
			s += "Mixture model encoding of categories with %i dimensions\n" % (self.D)
		s += "-> Prior distribution: %s\n" % self.prior_distribution.info()
		if self.use_decoder:
			s += "-> Decoder network: %s\n" % self.decoder.info()
		s += "\n".join(["-> [%i] " % (flow_index+1) + flow.info() for flow_index, flow in enumerate(self.flow_layers)])
		return s

## help_layers
class SimpleLinearLayer(nn.Module):

	def __init__(self, c_in, c_out, data_init=False):
		super().__init__()
		self.layer = nn.Linear(c_in, c_out)
		if data_init:
			scale_dims = int(c_out//2)
			self.layer.weight.data[scale_dims:,:] = 0
			self.layer.weight.data = self.layer.weight.data * 4 / np.sqrt(c_out/2)
			self.layer.bias.data.zero_()

	def forward(self, x, **kwargs):
		return self.layer(x)

	def initialize_zeros(self):
		self.layer.weight.data.zero_()
		self.layer.bias.data.zero_()

class LinearNet(nn.Module):

	def __init__(self, c_in, c_out, num_layers, hidden_size, ext_input_dims=0, zero_init=False):
		super().__init__()
		self.inp_layer = nn.Sequential(
				nn.Linear(c_in, hidden_size),
				nn.GELU()
			)
		self.main_net = []
		for i in range(num_layers):
			self.main_net += [
				nn.Linear(hidden_size if i>0 else hidden_size + ext_input_dims, 
						  hidden_size),
				nn.GELU()
			]
		self.main_net += [
			nn.Linear(hidden_size, c_out)
		]
		self.main_net = nn.Sequential(*self.main_net)
		if zero_init:
			self.main_net[-1].weight.data.zero_()
			self.main_net[-1].bias.data.zero_()

	def forward(self, x, ext_input=None, **kwargs):
		x_feat = self.inp_layer(x)
		if ext_input is not None:
			x_feat = torch.cat([x_feat, ext_input], dim=-1)
		out = self.main_net(x_feat)
		return out

	def set_bias(self, bias):
		self.main_net[-1].bias.data = bias

class DecoderLinear(nn.Module):
	"""
	A simple linear decoder with flexible number of layers. 
	"""

	def __init__(self, num_categories, embed_dim, hidden_size, num_layers, class_prior_log=None):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.layers = LinearNet(c_in=3*embed_dim, 
								c_out=num_categories,
								hidden_size=hidden_size,
								num_layers=num_layers)
		self.log_softmax = nn.LogSoftmax(dim=-1)

		if class_prior_log is not None:
			if isinstance(class_prior_log, np.ndarray):
				class_prior_log = torch.from_numpy(class_prior_log)
			self.layers.set_bias(class_prior_log)

	def forward(self, z_cont):
		z_cont = torch.cat([z_cont, F.elu(z_cont), F.elu(-z_cont)], dim=-1)
		out = self.layers(z_cont)
		logits = self.log_softmax(out)
		return logits

	def info(self):
		return "Linear model with hidden size %i and %i layers" % (self.hidden_size, self.num_layers)