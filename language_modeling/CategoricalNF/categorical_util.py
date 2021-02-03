import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

import numpy as np
import sys
import math
import os
from glob import glob

def general_args_to_params(args, model_params=None):

	optimizer_params = {
		"optimizer": args.optimizer,
		"learning_rate": args.learning_rate,
		"weight_decay": args.weight_decay,
		"lr_decay_factor": args.lr_decay_factor,
		"lr_decay_step": args.lr_decay_step,
		"lr_minimum": args.lr_minimum,
		"momentum": args.momentum,
		"beta1": args.beta1,
		"beta2": args.beta2,
		"warmup": args.warmup
	}

	return model_params, optimizer_params

def prior_distribution_args_to_params(args, add_name=""):
	prior_dist_params = {
		"distribution_type": getattr(args, "%sprior_dist_type" % add_name),
		"mu": getattr(args, "%sprior_dist_mu" % add_name),
		"sigma": getattr(args, "%sprior_dist_sigma" % add_name),
		"start_x": getattr(args, "%sprior_dist_start_x" % add_name),
		"stop_x": getattr(args, "%sprior_dist_stop_x" % add_name)
	}

	return prior_dist_params

def encoding_args_to_params(args, postfix=""):
	params = {
		"use_dequantization": getattr(args, "encoding_dequantization" + postfix),
		"use_variational": getattr(args, "encoding_variational" + postfix),
		"use_decoder": getattr(args, "encoding_use_decoder" + postfix),
		"num_dimensions": getattr(args, "encoding_dim" + postfix), 
		"flow_config": {
			"num_flows": getattr(args, "encoding_num_flows" + postfix),
			"hidden_layers": getattr(args, "encoding_hidden_layers" + postfix),
			"hidden_size": getattr(args, "encoding_hidden_size" + postfix)
		},
		"decoder_config": {
			"num_layers": getattr(args, "encoding_dec_num_layers" + postfix),
			"hidden_size": getattr(args, "encoding_dec_hidden_size" + postfix)
		}
	}
	return params

def scheduler_args_to_params(args, param_names):
	params = {}
	for name in param_names:
		params[name] = {
			p: getattr(args, "%s_%s" % (name, p)) for p in \
				["scheduler_type", "scheduler_end_val", "scheduler_start_val", "scheduler_step_size", "scheduler_logit", "scheduler_delay"]
		}
	return params

def args_to_params(args):
	model_params, optimizer_params = general_args_to_params(args, model_params=dict())
	model_params["prior_distribution"] = prior_distribution_args_to_params(args)
	model_params["categ_encoding"] = encoding_args_to_params(args)
	sched_params = scheduler_args_to_params(args, ["beta"])
	model_params.update(sched_params)
	dataset_params = {
		"max_seq_len": args.max_seq_len,
		"dataset": args.dataset,
		"use_rnn": args.use_rnn
	}
	coupling_params = {p_name: getattr(args, p_name) for p_name in vars(args) if p_name.startswith("coupling_")}
	model_params.update(coupling_params)
	model_params.update(dataset_params)
	return model_params, optimizer_params

def get_param_val(param_dict, key, default_val=None, allow_default=True, error_location="", warning_if_default=True):
	if key in param_dict:
		return param_dict[key]
	elif allow_default:
		if warning_if_default:
			print("[#] WARNING: Using default value %s for key %s" % (str(default_val), str(key)))
		return default_val
	else:
		assert False, "[!] ERROR (%s): could not find key \"%s\" in the dictionary although it is required." % (error_location, str(key))

def create_prior_distribution(distribution_params):
	distribution_type = get_param_val(distribution_params, "distribution_type", PriorDistribution.LOGISTIC)
	input_params = {key:val for key, val in distribution_params.items() if val is not None}

	if PriorDistribution.GAUSSIAN == distribution_type:
		return GaussianDistribution(**input_params)
	elif PriorDistribution.LOGISTIC == distribution_type:
		return LogisticDistribution(**input_params)
	else:
		print("[!] ERROR: Unknown distribution type %s" % str(distribution_type))
		sys.exit(1)

def create_scheduler(scheduler_params, param_name=None):
	sched_type = get_param_val(scheduler_params, "scheduler_type", allow_default=False)
	end_val = get_param_val(scheduler_params, "scheduler_end_val", allow_default=False)
	start_val = get_param_val(scheduler_params, "scheduler_start_val", allow_default=False)
	stepsize = get_param_val(scheduler_params, "scheduler_step_size", allow_default=False)
	logit = get_param_val(scheduler_params, "scheduler_logit", allow_default=False)
	delay = get_param_val(scheduler_params, "scheduler_delay", allow_default=False)

	if sched_type == "constant":
		return ConstantScheduler(const_val=end_val, param_name=param_name)
	elif sched_type == "linear":
		return LinearScheduler(start_val=start_val, end_val=end_val, stepsize=stepsize, delay=delay, param_name=param_name)
	elif sched_type == "sigmoid":
		return SigmoidScheduler(start_val=start_val, end_val=end_val, logit_factor=logit, stepsize=stepsize, delay=delay, param_name=param_name)
	elif sched_type == "exponential":
		return ExponentialScheduler(start_val=start_val, end_val=end_val, logit_factor=logit, stepsize=stepsize, delay=delay, param_name=param_name)
	else:
		print("[!] ERROR: Unknown scheduler type \"%s\"" % str(sched_type))
		sys.exit(1)

OPTIMIZER_SGD = 0
OPTIMIZER_ADAM = 1
OPTIMIZER_ADAMAX = 2
OPTIMIZER_RMSPROP = 3
OPTIMIZER_RADAM = 4
OPTIMIZER_ADAM_WARMUP = 5

def create_optimizer_from_args(parameters_to_optimize, optimizer_params):
	if optimizer_params["optimizer"] == OPTIMIZER_SGD:
		optimizer = torch.optim.SGD(parameters_to_optimize, 
									lr=optimizer_params["learning_rate"], 
									weight_decay=optimizer_params["weight_decay"],
									momentum=optimizer_params["momentum"])
	elif optimizer_params["optimizer"] == OPTIMIZER_ADAM:
		optimizer = torch.optim.Adam(parameters_to_optimize, 
									 lr=optimizer_params["learning_rate"],
									 betas=(optimizer_params["beta1"], optimizer_params["beta2"]),
									 weight_decay=optimizer_params["weight_decay"])
	elif optimizer_params["optimizer"] == OPTIMIZER_ADAMAX:
		optimizer = torch.optim.Adamax(parameters_to_optimize, 
									   lr=optimizer_params["learning_rate"],
									   weight_decay=optimizer_params["weight_decay"])
	elif optimizer_params["optimizer"] == OPTIMIZER_RMSPROP:
		optimizer = torch.optim.RMSprop(parameters_to_optimize, 
										lr=optimizer_params["learning_rate"],
										weight_decay=optimizer_params["weight_decay"])
	elif optimizer_params["optimizer"] == OPTIMIZER_RADAM:
		optimizer = RAdam(parameters_to_optimize, 
						  lr=optimizer_params["learning_rate"],
						  betas=(optimizer_params["beta1"], optimizer_params["beta2"]),
						  weight_decay=optimizer_params["weight_decay"])
	elif optimizer_params["optimizer"] == OPTIMIZER_ADAM_WARMUP:
		optimizer = AdamW(parameters_to_optimize, 
						  lr=optimizer_params["learning_rate"],
						  weight_decay=optimizer_params["weight_decay"],
						  betas=(optimizer_params["beta1"], optimizer_params["beta2"]),
						  warmup=optimizer_params["warmup"])
	else:
		print("[!] ERROR: Unknown optimizer: " + str(optimizer_params["optimizer"]))
		sys.exit(1)
	return optimizer

def load_model(checkpoint_path, model=None, optimizer=None, lr_scheduler=None, load_best_model=False, warn_unloaded_weights=True):
	# Determine the checkpoint file to load
	if os.path.isdir(checkpoint_path):
		checkpoint_files = sorted(glob(os.path.join(checkpoint_path, "*.tar")))
		if len(checkpoint_files) == 0:
			print("No checkpoint files found at", checkpoint_path)
			return dict()
		checkpoint_file = checkpoint_files[-1]
	else:
		checkpoint_file = checkpoint_path

	# Loading checkpoint
	print("Loading checkpoint \"" + str(checkpoint_file) + "\"")
	if torch.cuda.is_available():
		checkpoint = torch.load(checkpoint_file)
	else:
		checkpoint = torch.load(checkpoint_file, map_location='cpu')
	
	# If best model should be loaded, look for it if checkpoint_path is a directory
	if os.path.isdir(checkpoint_path) and load_best_model:
		if os.path.isfile(checkpoint["best_save_dict"]["file"]):
			print("Load best model!")
			return load_model(checkpoint["best_save_dict"]["file"], model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, load_best_model=False)
		else:
			print("[!] WARNING: Best save dict file is listed as \"%s\", but file could not been found. Using default one..." % checkpoint["best_save_dict"]["file"])

	# Load the model parameters
	if model is not None:
		pretrained_model_dict = {key: val for key, val in checkpoint['model_state_dict'].items()}
		model_dict = model.state_dict()
		unchanged_keys = [key for key in model_dict.keys() if key not in pretrained_model_dict.keys()]
		if warn_unloaded_weights and len(unchanged_keys) != 0: # Parameters in this list might have been forgotten to be saved
			print("[#] WARNING: Some weights have been left unchanged by the loading of the model: " + str(unchanged_keys))
		model_dict.update(pretrained_model_dict)
		model.load_state_dict(model_dict)
	# Load the state and parameters of the optimizer
	if optimizer is not None and 'optimizer_state_dict' in checkpoint:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# Load the state of the learning rate scheduler
	if lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
		lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	# Load the additional parameters that were saved in the dict
	add_param_dict = dict()
	for key, val in checkpoint.items():
		if "state_dict" not in key:
			add_param_dict[key] = val
	return add_param_dict

class Tracker:

	def __init__(self, exp_decay=1.0):
		self.val_sum = 0.0 
		self.counter = 0
		self.exp_decay = exp_decay

	def add(self, val):
		self.val_sum = self.val_sum * self.exp_decay + val 
		self.counter = self.counter * self.exp_decay + 1

	def get_mean(self, reset=False):
		if self.counter <= 0:
			mean = 0
		else:
			mean = self.val_sum / self.counter
		if reset:
			self.reset()
		return mean

	def reset(self):
		self.val_sum = 0.0
		self.counter = 0

def _create_length_mask(length, max_len=None, dtype=torch.float32):
	if max_len is None:
		max_len = length.max()
	mask = (torch.arange(max_len, device=length.device).view(1, max_len) < length.unsqueeze(dim=-1)).to(dtype=dtype)
	return mask

def create_transformer_mask(length, max_len=None, dtype=torch.float32):
	mask = _create_length_mask(length=length, max_len=max_len, dtype=torch.bool)
	mask = ~mask # Negating mask, as positions that should be masked, need a True, and others False
	# mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask

def create_channel_mask(length, max_len=None, dtype=torch.float32):
	mask = _create_length_mask(length=length, max_len=max_len, dtype=dtype)
	mask = mask.unsqueeze(dim=-1) # Unsqueeze over channels
	return mask

def write_dict_to_tensorboard(writer, val_dict, base_name, iteration):
	for name, val in val_dict.items():
		if isinstance(val, dict):
			write_dict_to_tensorboard(writer, val, base_name=base_name+"/"+name, iteration=iteration)
		elif isinstance(val, (list, np.ndarray)):
			continue
		elif isinstance(val, (int, float)):
			writer.add_scalar(base_name + "/" + name, val, iteration)

def append_in_dict(val_dict, key, new_val):
	if key not in val_dict:
		val_dict[key] = list()
	val_dict[key].append(new_val)

def one_hot(x, num_classes, dtype=torch.float32):
	if isinstance(x, np.ndarray):
		x_onehot = np.zeros(x.shape + (num_classes,), dtype=np.float32)
		x_onehot[np.arange(x.shape[0]), x] = 1.0
	elif isinstance(x, torch.Tensor):
		assert torch.max(x) < num_classes, "[!] ERROR: One-hot input has larger entries (%s) than classes (%i)" % (str(torch.max(x)), num_classes)
		x_onehot = x.new_zeros(x.shape + (num_classes,), dtype=dtype)
		x_onehot.scatter_(-1, x.unsqueeze(dim=-1), 1)
	else:
		print("[!] ERROR: Unknown object given for one-hot conversion:", x)
		sys.exit(1)
	return x_onehot

def create_T_one_hot(length, dataset_max_len, dtype=torch.float32):
	if length is None:
		print("Length", length)
		print("Dataset max len", dataset_max_len)
	max_batch_len = length.max()
	assert max_batch_len <= dataset_max_len, "[!] ERROR - T_one_hot: Max batch size (%s) was larger than given dataset max length (%s)" % (str(max_batch_len.item()), str(dataset_max_len))
	time_range = torch.arange(max_batch_len, device=length.device).view(1, max_batch_len).expand(length.size(0),-1)
	length_onehot_pos = one_hot(x=time_range, num_classes=dataset_max_len, dtype=dtype)
	inv_time_range = (length.unsqueeze(dim=-1)-1) - time_range
	length_mask = (inv_time_range>=0.0).float()
	inv_time_range = inv_time_range.clamp(min=0.0)
	length_onehot_neg = one_hot(x=inv_time_range, num_classes=dataset_max_len, dtype=dtype)
	length_onehot = torch.cat([length_onehot_pos, length_onehot_neg], dim=-1)
	length_onehot = length_onehot * length_mask.unsqueeze(dim=-1)
	return length_onehot

def run_padded_LSTM(x, lstm_cell, length, input_memory=None, return_final_states=False):
	if length is not None and (length != x.size(1)).sum() > 0:
		# Sort input elements for efficient LSTM application
		sorted_lengths, perm_index = length.sort(0, descending=True)
		x = x[perm_index]

		packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, sorted_lengths.cpu(), batch_first=True)
		packed_outputs, _ = lstm_cell(packed_input, input_memory)
		outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

		# Redo sort
		_, unsort_indices = perm_index.sort(0, descending=False)
		outputs = outputs[unsort_indices]
	else:
		outputs, _ = lstm_cell(x, input_memory)
	return outputs

def run_sequential_with_mask(net, x, length=None, channel_padding_mask=None, src_key_padding_mask=None, length_one_hot=None, time_embed=None, gt=None, importance_weight=1, detail_out=False, **kwargs):
	dict_detail_out = dict()
	if channel_padding_mask is None:
		nn_out = net(x)
	else:
		x = x * channel_padding_mask
		for l in net:
			x = l(x)
		nn_out = x * channel_padding_mask # Making sure to zero out the outputs for all padding symbols

	if not detail_out:
		return nn_out
	else:
		return nn_out, dict_detail_out

## PriorDistribution
class PriorDistribution(nn.Module):

	GAUSSIAN = 0
	LOGISTIC = 1

	def __init__(self, **kwargs):
		super().__init__()
		self.distribution = self._create_distribution(**kwargs)

	def _create_distribution(self, **kwargs):
		raise NotImplementedError

	def forward(self, shape=None):
		return self.sample(shape=shape)

	def sample(self, shape=None):
		if shape is None:
			return self.distribution.sample()
		else:
			return self.distribution.sample(sample_shape=shape)

	def log_prob(self, x):
		logp = self.distribution.log_prob(x)
		assert torch.isnan(logp).sum() == 0, "[!] ERROR: Found NaN values in log-prob of distribution.\n" + \
				"NaN logp: " + str(torch.isnan(logp).sum().item()) + "\n" + \
				"NaN x: " + str(torch.isnan(x).sum().item()) + ", X(abs) max: " + str(x.abs().max())
		return logp

	def prob(self, x):
		return self.log_prob(x).exp()

	def icdf(self, x):
		assert ((x < 0) | (x > 1)).sum() == 0, \
			   "[!] ERROR: Found values outside the range of 0 to 1 as input to the inverse cumulative distribution function."
		return self.distribution.icdf(x)

	def cdf(self, x):
		return self.distribution.cdf(x)

	def info(self):
		raise NotImplementedError

	@staticmethod
	def get_string_of_distributions():
		return "%i - Gaussian, %i - Logistic" % (PriorDistribution.GAUSSIAN, PriorDistribution.LOGISTIC)

class GaussianDistribution(PriorDistribution):

	def __init__(self, mu=0.0, sigma=1.0, **kwargs):
		super().__init__(mu=mu, sigma=sigma, **kwargs)
		self.mu = mu
		self.sigma = sigma

	def _create_distribution(self, mu=0.0, sigma=1.0, **kwargs):
		return torch.distributions.normal.Normal(loc=mu, scale=sigma)

	def info(self):
		return "Gaussian distribution with mu=%f and sigma=%f" % (self.mu, self.sigma)

class LogisticDistribution(PriorDistribution):

	def __init__(self, mu=0.0, sigma=1.0, eps=1e-4, **kwargs):
		sigma = sigma / 1.81 # STD of a logistic distribution is about 1.81 in default settings
		super().__init__(mu=mu, sigma=sigma)
		self.mu = mu
		self.sigma = sigma
		self.log_sigma = np.log(self.sigma)
		self.eps = eps

	def _create_distribution(self, mu=0.0, sigma=1.0, **kwargs):
		return torch.distributions.uniform.Uniform(low=0.0, high=1.0)

	def _safe_log(self, x, eps=1e-22):
		return torch.log(x.clamp(min=1e-22))

	def _shift_x(self, x):
		return LogisticDistribution.shift_x(x, self.mu, self.sigma, self.log_sigma)

	def _unshift_x(self, x):
		return LogisticDistribution.unshift_x(x, self.mu, self.sigma, self.log_sigma)

	@staticmethod
	def shift_x(x, mu, sigma, log_sigma=None):
		if log_sigma is None:
			log_sigma = sigma.log()
		x = x.double()
		z = -torch.log(x.reciprocal() - 1.)
		ldj = -torch.log(x) - torch.log(1. - x)
		z, ldj = z.float(), ldj.float()
		z = z * sigma + mu
		ldj = ldj - log_sigma
		return z, ldj

	@staticmethod
	def unshift_x(x, mu, sigma, log_sigma=None):
		if log_sigma is None:
			log_sigma = sigma.log()
		x = (x - mu) / sigma
		z = torch.sigmoid(x)
		ldj = F.softplus(x) + F.softplus(-x) + log_sigma
		return z, ldj

	def sample(self, shape=None, return_ldj=False, temp=1.0):
		samples = super().sample(shape=shape)
		if shape[-1] != 1:
			samples = samples.squeeze(dim=-1)
		samples = ( samples * (1-self.eps) ) + self.eps/2
		if temp == 1.0:
			samples, sample_ldj = self._shift_x(samples)
		else:
			samples, sample_ldj = SigmoidUniformDistribution.shift_x(samples, self.mu, self.sigma*temp, self.log_sigma + np.log(temp))
		if not return_ldj:
			return samples
		else:
			return samples, sample_ldj

	def log_prob(self, x):
		x, ldj = self._unshift_x(x)
		# Distribution logp not needed as it is 0 anyways
		logp = - ldj

		assert torch.isnan(logp).sum() == 0, "[!] ERROR: Found NaN values in log-prob of distribution.\n" + \
				"NaN logp: " + str(torch.isnan(logp).sum().item()) + "\n" + \
				"NaN x: " + str(torch.isnan(x).sum().item()) + ", X(abs) max: " + str(x.abs().max())

		return logp

	def icdf(self, x, return_ldj=False):
		assert ((x < 0) | (x > 1)).sum() == 0, \
			   "[!] ERROR: Found values outside the range of 0 to 1 as input to the inverse cumulative distribution function."
		z, ldj = self._shift_x(x)
		if not return_ldj:
			return z
		else:
			return z, ldj

	def cdf(self, x, return_ldj=False):
		z, ldj = self._unshift_x(x)
		if not return_ldj:
			return z
		else:
			return z, ldj

	def info(self):
		return "Sigmoid Uniform distribution with mu=%.2f and sigma=%.2f" % (self.mu, self.sigma)

## beta scheduler
class ParameterScheduler:

	def __init__(self, param_name=None):
		self.param_name = param_name

	def get(self, iteration):
		raise NotImplementedError

	def info(self):
		return self._scheduler_description() + \
			   " for parameter %s" % str(self.param_name) if self.param_name is not None else ""

	def _scheduler_description(self):
		raise NotImplementedError

class ConstantScheduler(ParameterScheduler):

	def __init__(self, const_val, param_name=None):
		super().__init__(param_name=param_name)
		self.const_val = const_val

	def get(self, iteration):
		return self.const_val

	def _scheduler_description(self):
		return "Constant Scheduler on value %s" % str(self.const_val)

class SlopeScheduler(ParameterScheduler):

	def __init__(self, start_val, end_val, stepsize, logit_factor=0, delay=0, param_name=None):
		super().__init__(param_name=param_name)
		self.start_val = start_val
		self.end_val = end_val
		self.logit_factor = logit_factor
		self.stepsize = stepsize
		self.delay = delay
		assert self.stepsize > 0

	def get(self, iteration):
		if iteration < self.delay:
			return self.start_val
		else:
			iteration = iteration - self.delay
			return self.get_val(iteration)

	def get_val(self, iteration):
		raise NotImplementedError

class SigmoidScheduler(SlopeScheduler):

	def __init__(self, start_val, end_val, logit_factor, stepsize, delay=0, param_name=None):
		super().__init__(start_val=start_val, 
						 end_val=end_val, 
						 logit_factor=logit_factor, 
						 stepsize=stepsize, 
						 delay=delay, 
						 param_name=param_name)

	def get_val(self, iteration):
		return self.start_val + (self.end_val - self.start_val) / (1.0 + np.exp(-self.logit_factor * (iteration-self.stepsize)))

	def _scheduler_description(self):
		return "Sigmoid Scheduler from %s to %s with logit factor %s and stepsize %s" % \
				(str(self.start_val), str(self.end_val), str(self.logit_factor), str(self.stepsize))

class LinearScheduler(SlopeScheduler):

	def __init__(self, start_val, end_val, stepsize, delay=0, param_name=None):
		super().__init__(start_val=start_val, 
						 end_val=end_val, 
						 logit_factor=0, 
						 stepsize=stepsize, 
						 delay=delay, 
						 param_name=param_name)

	def get_val(self, iteration):
		if iteration >= self.stepsize:
			return self.end_val
		else:
			return self.start_val + (self.end_val - self.start_val) * (iteration * 1.0 / self.stepsize)

	def _scheduler_description(self):
		return "Linear Scheduler from %s to %s in %s steps" % \
				(str(self.start_val), str(self.end_val), str(self.stepsize))

class ExponentialScheduler(SlopeScheduler):

	def __init__(self, start_val, end_val, logit_factor, stepsize, delay=0, param_name=None):
		super().__init__(start_val=start_val, 
						 end_val=end_val, 
						 logit_factor=logit_factor, 
						 stepsize=stepsize, 
						 delay=delay, 
						 param_name=param_name)

	def get_val(self, iteration):
		return self.start_val + (self.end_val - self.start_val) * (1 - self.logit_factor ** (-iteration*1.0/self.stepsize))

	def _scheduler_description(self):
		return "Exponential Scheduler from %s to %s with logit %s and stepsize %s" % \
				(str(self.start_val), str(self.end_val), str(self.logit_factor), str(self.stepsize))

## optimizer
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

## flow model related
def create_embed_layer(vocab, vocab_size, default_embed_layer_dims):
	## Creating an embedding layer either from a torchtext vocabulary or from scratch
	use_vocab_vectors = (vocab is not None and vocab.vectors is not None)
	embed_layer_dims = vocab.vectors.shape[1] if use_vocab_vectors else default_embed_layer_dims
	vocab_size = len(vocab) if use_vocab_vectors else vocab_size
	embed_layer = nn.Embedding(vocab_size, embed_layer_dims)
	if use_vocab_vectors:
		embed_layer.weight.data.copy_(vocab.vectors)
		embed_layer.weight.requires_grad = True
	return embed_layer, vocab_size

def safe_log(x):
	return torch.log(x.clamp(min=1e-22))

def _log_pdf(x, mean, log_scale):
	"""Element-wise log density of the logistic distribution."""
	z = (x - mean) * torch.exp(-log_scale)
	log_p = z - log_scale - 2 * F.softplus(z)

	return log_p

def _log_cdf(x, mean, log_scale):
	"""Element-wise log CDF of the logistic distribution."""
	z = (x - mean) * torch.exp(-log_scale)
	log_p = F.logsigmoid(z)

	return log_p

def mixture_log_pdf(x, prior_logits, means, log_scales):
	"""Log PDF of a mixture of logistic distributions."""
	log_ps = F.log_softmax(prior_logits, dim=-1) \
		+ _log_pdf(x.unsqueeze(dim=-1), means, log_scales)
	log_p = torch.logsumexp(log_ps, dim=-1)

	return log_p

def mixture_log_cdf(x, prior_logits, means, log_scales):
	"""Log CDF of a mixture of logistic distributions."""
	log_ps = F.log_softmax(prior_logits, dim=-1) \
		+ _log_cdf(x.unsqueeze(dim=-1), means, log_scales)
	log_p = torch.logsumexp(log_ps, dim=-1)

	return log_p

def mixture_inv_cdf(y, prior_logits, means, log_scales,
            		eps=1e-10, max_iters=100):
	# Inverse CDF of a mixture of logisitics. Iterative algorithm.
	if y.min() <= 0 or y.max() >= 1:
		raise RuntimeError('Inverse logisitic CDF got y outside (0, 1)')

	def body(x_, lb_, ub_):
		cur_y = torch.exp(mixture_log_cdf(x_, prior_logits, means,
		                                  log_scales))
		gt = (cur_y > y).type(y.dtype)
		lt = 1 - gt
		new_x_ = gt * (x_ + lb_) / 2. + lt * (x_ + ub_) / 2.
		new_lb = gt * lb_ + lt * x_
		new_ub = gt * x_ + lt * ub_
		return new_x_, new_lb, new_ub

	x = torch.zeros_like(y)
	max_scales = torch.sum(torch.exp(log_scales), dim=-1, keepdim=True)
	lb, _ = (means - 20 * max_scales).min(dim=-1)
	ub, _ = (means + 20 * max_scales).max(dim=-1)
	diff = float('inf')

	i = 0
	while diff > eps and i < max_iters:
		new_x, lb, ub = body(x, lb, ub)
		diff = (new_x - x).abs().max()
		x = new_x
		i += 1

	return x

def inverse(x, reverse=False):
	"""Inverse logistic function."""
	if reverse:
		z = torch.sigmoid(x)
		ldj = F.softplus(x) + F.softplus(-x)
	else:
		z = -safe_log(x.reciprocal() - 1.)
		ldj = -safe_log(x) - safe_log(1. - x)

	return z, ldj