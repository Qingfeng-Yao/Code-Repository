import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.linalg
from collections import defaultdict 
import math

from utils.misc import create_T_one_hot, run_padded_LSTM, run_sequential_with_mask, mixture_log_cdf, mixture_log_pdf, inverse_func, safe_log, mixture_inv_cdf

class TimeConcat(nn.Module):
    	
	def __init__(self, time_embed, input_dp_rate=0.0):
		super().__init__()
		self.time_embed_layer = time_embed
		self.input_dropout = InputDropout(input_dp_rate)

	def forward(self, x, time_embed=None, length_one_hot=None, length=None):
		if time_embed is None:
			if length_one_hot is None and length is not None:
				length_one_hot = create_T_one_hot(length, dataset_max_len=int(self.time_embed_layer.weight.data.shape[1]//2))
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

		if input_dp_rate < 1.0:
			time_embed = nn.Linear(2*max_seq_len, int(hidden_size//8))
			time_embed_dim = time_embed.weight.data.shape[0]
			self.time_concat = TimeConcat(time_embed=time_embed, input_dp_rate=input_dp_rate)
		else:
			self.time_concat = None
			time_embed_dim = 0

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
		if self.time_concat is not None:
			_inp_embed = self.time_concat(x=_inp_embed, length_one_hot=length_one_hot, length=length)
		embed = torch.cat([_inp_embed.new_zeros(_inp_embed.size(0),1,_inp_embed.size(2)), _inp_embed[:,:-1]], dim=1)

		lstm_out = run_padded_LSTM(x=embed, lstm_cell=self.lstm_module, length=length.cpu())

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
			nn_out = run_sequential_with_mask(self.nn, x, length=length, **kwargs)
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
			z_out = mixture_log_cdf(x=orig_z, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s).exp()

			# Regularize mixtures if wanted (only done during training as LDJ is increased)
			if reg_max > 0 and is_training:
				reg_ldj = torch.stack([safe_log(z_out), safe_log(1-z_out)], dim=-1)/np.log(10) # Change to 10 base
				reg_ldj = reg_ldj.clamp(max=-reg_max) + reg_max
				reg_ldj = reg_ldj.sum(dim=-1)
				reg_ldj = reg_ldj * change_mask
			else:
				reg_ldj = torch.zeros_like(z_out)

			# Map from [0,1] domain back to [-inf,inf] by inverse sigmoid
			z_out, mixt_ldj = inverse_func(z_out)
			# Output affine transformation
			z_out = (z_out + t) * log_s.exp()
			# Determine LDJ of the transformation
			logistic_ldj = mixture_log_pdf(orig_z, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s)
			# Combine all LDJs
			ldj = (change_mask * (log_s + mixt_ldj + logistic_ldj + reg_ldj * reg_factor)).sum(dim=[1,2])
		else:
			# Reverse output affine transformation
			z_out = orig_z * (-log_s).exp() - t
			# Apply sigmoid to map back to [0,1] domain
			z_out, mixt_ldj = inverse_func(z_out, reverse=True)
			# Clamping to prevent numerical instabilities
			z_out = z_out.clamp(1e-5, 1. - 1e-5)
			# Inverse the cummulative distribution function of mixtures. Iterative algorithm, maps back to [-inf,inf]
			z_out = mixture_inv_cdf(z_out, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s)
			# Determine LDJ of this transformation
			logistic_ldj = mixture_log_pdf(z_out, prior_logits=log_pi, means=mixt_t, log_scales=mixt_log_s)
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