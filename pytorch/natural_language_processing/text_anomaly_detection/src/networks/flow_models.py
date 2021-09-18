import torch
import torch.nn as nn

import numpy as np
import scipy.linalg
from collections import defaultdict 

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