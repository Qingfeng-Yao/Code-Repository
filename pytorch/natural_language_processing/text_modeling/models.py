import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
from statistics import mean
import os
import datetime
import json

from utils import get_param_val, WrappedDataParallel, get_device, debug_level, append_in_dict, create_channel_mask, create_optimizer_from_args, Tracker, write_dict_to_tensorboard, load_model
from distributions import create_prior_distribution
from scheduler import create_scheduler

from datasets.text8 import Text8Dataset
from datasets.penntreebank import PennTreeBankDataset

from rnn import LSTMModel
from cnf import CNFLanguageModeling
from df import DFModel

class TaskTemplate:
	
	def __init__(self, model, model_params, name, load_data=True, debug=False, batch_size=64, drop_last=False, num_workers=None):
		# Saving parameters
		self.name = name 
		self.model = model
		self.model_params = model_params
		self.batch_size = batch_size
		self.train_batch_size = batch_size
		self.debug = debug

		# Initializing dataset parameters
		self.train_dataset = None 
		self.val_dataset = None 
		self.test_dataset = None
		self.train_epoch = 0
		
		# Load data if specified, and create data loaders
		if load_data:
			self._load_datasets()
			self._initialize_data_loaders(drop_last=drop_last, num_workers=num_workers)
		else:
			self.train_data_loader = None 
			self.train_data_loader_iter = None
			self.val_data_loader = None 
			self.test_data_loader = None

		# Create a dictionary to store summary metrics in
		self.summary_dict = {}

		# Placeholders for visualization
		self.gen_batch = None
		self.class_colors = None

		# Put model on correct device
		self.model.to(get_device())


	def _initialize_data_loaders(self, drop_last, num_workers):
		if num_workers is None:
			if isinstance(self.model, nn.DataParallel) and torch.cuda.device_count() > 1:
				num_workers = torch.cuda.device_count()
			else:
				num_workers = 1

		def _init_fn(worker_id):
			np.random.seed(42)
		# num_workers = 1
		# Initializes all data loaders with the loaded datasets
		if hasattr(self.train_dataset, "get_sampler"):
			self.train_data_loader = data.DataLoader(self.train_dataset, batch_sampler=self.train_dataset.get_sampler(self.train_batch_size, drop_last=drop_last), pin_memory=True, 
													 num_workers=num_workers, worker_init_fn=_init_fn)
			self.val_data_loader = data.DataLoader(self.val_dataset, batch_sampler=self.val_dataset.get_sampler(self.train_batch_size, drop_last=False), pin_memory=True, num_workers=1, worker_init_fn=_init_fn)
			self.test_data_loader = data.DataLoader(self.test_dataset, batch_sampler=self.test_dataset.get_sampler(self.train_batch_size, drop_last=False), pin_memory=True, num_workers=1, worker_init_fn=_init_fn)		
		else:
			self.train_data_loader = data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True, drop_last=drop_last, num_workers=num_workers,
													 worker_init_fn=_init_fn)
			self.val_data_loader = data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=1, worker_init_fn=_init_fn)
			self.test_data_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=1, worker_init_fn=_init_fn)
		self.train_data_loader_iter = iter(self.train_data_loader) # Needed to retrieve batch by batch from dataset
		

	def train_step(self, iteration=0):
		# Check if training data was correctly loaded
		if self.train_data_loader_iter is None:
			print("[!] ERROR: Iterator of the training data loader was None. Additional parameters: " + \
				  "train_data_loader was %sloaded, " % ("not " if self.train_data_loader is None else "") + \
				  "train_dataset was %sloaded." % ("not " if self.train_dataset is None else ""))
		
		# Get batch and put it on correct device
		batch = self._get_next_batch()
		batch = TaskTemplate.batch_to_device(batch)

		# Perform task-specific training step
		return self._train_batch(batch, iteration=iteration)


	def eval(self, data_loader=None, **kwargs):
		# Default: if no dataset is specified, we use validation dataset
		if data_loader is None:
			assert self.val_data_loader is not None, "[!] ERROR: Validation dataset not loaded. Please load the dataset beforehand for evaluation."
			data_loader = self.val_data_loader
		is_test = (data_loader == self.test_data_loader)

		start_time = time.time()
		torch.cuda.empty_cache()
		self.model.eval()
		
		# Prepare metrics
		total_nll, embed_nll, nll_counter = 0, 0, 0

		# Evaluation loop
		with torch.no_grad():
			for batch_ind, batch in enumerate(data_loader):
				if debug_level() == 0:
					print("Evaluation process: %4.2f%%" % (100.0 * batch_ind / len(data_loader)), end="\r")
				# Put batch on correct device
				batch = TaskTemplate.batch_to_device(batch)
				# Evaluate single batch
				batch_size = batch[0].size(0) if isinstance(batch, tuple) else batch.size(0)
				batch_nll = self._eval_batch(batch, is_test=is_test)
				total_nll += batch_nll.item() * batch_size
				nll_counter += batch_size

				if self.debug and batch_ind > 10:
					break

		avg_nll = total_nll / max(1e-5, nll_counter)
		detailed_metrics = {
			"negative_log_likelihood": avg_nll,
			"bpd": self.loss_to_bpd(avg_nll), # Bits per dimension
		}

		with torch.no_grad():
			self._eval_finalize_metrics(detailed_metrics, is_test=is_test, **kwargs)

		self.model.train()
		eval_time = int(time.time() - start_time)
		print("Finished %s with bpd of %4.3f (%imin %is)" % ("testing" if data_loader == self.test_data_loader else "evaluation", detailed_metrics["bpd"], eval_time/60, eval_time%60))
		torch.cuda.empty_cache()

		if "loss_metric" in detailed_metrics:
			loss_metric = detailed_metrics["loss_metric"]
		else:
			loss_metric = avg_nll
		
		return loss_metric, detailed_metrics


	def loss_to_bpd(self, loss):
		return (np.log2(np.exp(1)) * loss)


	def test(self, **kwargs):
		return self.eval(data_loader=self.test_data_loader, **kwargs)


	def add_summary(self, writer, iteration, checkpoint_path=None):
		# Adding metrics collected during training to the tensorboard
		# Function can/should be extended if needed
		for key, val in self.summary_dict.items():
			summary_key = "train_%s/%s" % (self.name, key)
			if not isinstance(val, list): # If it is not a list, it is assumably a single scalar
				writer.add_scalar(summary_key, val, iteration)
				self.summary_dict[key] = 0.0
			elif len(val) == 0: # Skip an empty list
				continue
			elif not isinstance(val[0], list): # For a list of scalars, report the mean
				writer.add_scalar(summary_key, mean(val), iteration)
				self.summary_dict[key] = list()
			else: # List of lists indicates a histogram
				val = [v for sublist in val for v in sublist]
				writer.add_histogram(summary_key, np.array(val), iteration)
				self.summary_dict[key] = list()


	def _ldj_per_layer_to_summary(self, ldj_per_layer, pre_phrase="ldj_layer_"):
		for layer_index, layer_ldj in enumerate(ldj_per_layer):
			if isinstance(layer_ldj, tuple) or isinstance(layer_ldj, list):
				for i in range(len(layer_ldj)):
					append_in_dict(self.summary_dict, "ldj_layer_%i_%i" % (layer_index, i), layer_ldj[i].detach().mean().item())
			elif isinstance(layer_ldj, dict):
				for key, ldj_val in layer_ldj.items():
					append_in_dict(self.summary_dict, "ldj_layer_%i_%s" % (layer_index, key), ldj_val.detach().mean().item())
			else:
				append_in_dict(self.summary_dict, "ldj_layer_%i" % layer_index, layer_ldj.detach().mean().item())


	def _get_next_batch(self):
		# Try to get next batch. If one epoch is over, the iterator throws an error, and we start a new iterator
		try:
			batch = next(self.train_data_loader_iter)
		except StopIteration:
			self.train_data_loader_iter = iter(self.train_data_loader)
			batch = next(self.train_data_loader_iter)
			self.train_epoch += 1
		return batch


	#######################################################
	### Abstract method to be implemented by subclasses ###
	#######################################################


	def _load_datasets(self):
		# Function for initializing datasets. Should set the following class parameters:
		# -> self.train_dataset
		# -> self.val_dataset
		# -> self.test_dataset
		raise NotImplementedError	


	def _train_batch(self, batch, iteration=0):
		# Given a batch, return the loss to be trained on
		raise NotImplementedError


	def _eval_batch(self, batch, is_test=False, take_mean=True):
		# Given a batch, return the negative log likelihood for its elements
		# Input arguments:
		# -> "is_test": True during testing, False during validation
		# -> "take_mean": If true, the return should be the average log 
		# 				  likelihood of the batch. Otherwise, return per element.
		raise NotImplementedError


	def finalize_summary(self, writer, iteration, checkpoint_path):
		# This function is called after the finishing training and performing testing.
		# Can be used to add something to the summary before finishing.
		pass


	def export_best_results(self, checkpoint_path, iteration):
		# This function is called if the last evaluation has been the best so far.
		# Can be used to add some output to the checkpoint directory.
		pass


	def initialize(self):
		# This function is called before starting the training.
		pass


	def _eval_finalize_metrics(self, detailed_metrics, is_test=False, initial_eval=False):
		# This function is called after evaluation finished. 
		# Can be used to add final metrics to the dictionary, which is added to the tensorboard.
		pass


	####################
	## Static methods ##
	####################

	@staticmethod
	def batch_to_device(batch): # batch is a list
		if isinstance(batch, tuple) or isinstance(batch, list):
			# print(len(batch)) # 2
			# print(batch[0].device, batch[1].device) # cpu
			batch = tuple([b.to(get_device()) for b in batch])
			# print(batch[0].device, batch[1].device) # cuda
			# print(batch[0].shape, batch[1].shape) # torch.Size([128, 288]) torch.Size([128])
			# print(batch[1][0], type(batch[1][0]), batch[1][0].device) type/<class 'torch.Tensor'>, device/cuda
		else:
			batch = batch.to(get_device())
		return batch

class TaskLanguageModeling(TaskTemplate):
	
	def __init__(self, model, model_params, load_data=True, debug=False, batch_size=64):
		super().__init__(model, model_params, load_data=load_data, debug=debug, batch_size=batch_size, name="TaskLanguageModeling")

		prior_dist_params = get_param_val(self.model_params, "prior_distribution", allow_default=False, error_location="TaskLanguageModeling - init")
		self.prior_distribution = create_prior_distribution(prior_dist_params)

		self.beta_scheduler = create_scheduler(self.model_params["beta"], "beta")
		
		self.summary_dict = {"log_prob": list(), "ldj": list(), "z": list(),
							 "beta": 0}


	def _load_datasets(self):
		self.max_seq_len = get_param_val(self.model_params, "max_seq_len", allow_default=False)

		dataset_name = get_param_val(self.model_params, "dataset", default_val="penntreebank")
		self.dataset_class = TaskLanguageModeling.get_dataset_class(dataset_name)
		print("Loading dataset %s..." % dataset_name)

		self.train_dataset = self.dataset_class(max_seq_len=self.max_seq_len, train=True)
		self.val_dataset = self.dataset_class(max_seq_len=self.max_seq_len, val=True)
		self.test_dataset = self.dataset_class(max_seq_len=self.max_seq_len, test=True)

		if hasattr(self.dataset_class, "get_length_prior"):
			# print("use length_prior!")
			self.length_prior = self.dataset_class.get_length_prior(max_seq_len=self.max_seq_len)
		else:
			# print("not use length_prior!")
			self.length_prior = None
		

	@staticmethod
	def get_dataset_class(dataset_name):
		if dataset_name == "penntreebank":
			dataset_class = PennTreeBankDataset
		elif dataset_name == "text8":
			dataset_class = Text8Dataset
		else:
			assert False, "[!] ERROR: Unknown dataset class \"%s\"" % (dataset_name)
		return dataset_class
		


	def _train_batch(self, batch, iteration=0):
		x_in, x_length, x_channel_mask = self._preprocess_batch(batch)
		if isinstance(self.model, LSTMModel):
			return self._train_batch_rnn(x_in, x_length, x_channel_mask)
		elif isinstance(self.model, DFModel):
			return self._train_batch_df(x_in, x_length, x_channel_mask)
		elif isinstance(self.model, CNFLanguageModeling):
			return self._train_batch_flow(x_in, x_length, x_channel_mask, iteration=iteration)

	def _train_batch_rnn(self, x_in, x_length, x_channel_mask, **kwargs):
		logprob, details = self.model(x_in, reverse=False, length=x_length, channel_padding_mask=x_channel_mask)
		self.summary_dict["log_prob"].append(-logprob.mean().item())
		self._ldj_per_layer_to_summary([details])
		loss, _, _ = self._calc_loss(torch.zeros_like(logprob), -logprob, x_length)
		return loss

	def _train_batch_df(self, x_in, x_length, x_channel_mask, **kwargs):
		vocab_dict = self.dataset_class.get_vocabulary()
		x = F.one_hot(x_in, num_classes = len(vocab_dict)).float()
		z = self.model(x, reverse=False)

		base_log_probs_sm = torch.nn.functional.log_softmax(self.model.base_log_probs, dim=-1).to(x_in.device)
		neglog_prob = -(z*base_log_probs_sm*x_channel_mask).sum(dim=[1,2])
		loss, _, _ = self._calc_loss(torch.zeros_like(neglog_prob), neglog_prob, x_length)
		return loss

	def _train_batch_flow(self, x_in, x_length, x_channel_mask, iteration=0, **kwargs):
		z, ldj, ldj_per_layer = self.model(x_in, reverse=False, get_ldj_per_layer=True, 
										   beta=self.beta_scheduler.get(iteration),
										   length=x_length)
		neglog_prob = -(self.prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
		neg_ldj = -ldj
		
		loss, neg_ldj, neglog_prob = self._calc_loss(neg_ldj, neglog_prob, x_length)

		self.summary_dict["log_prob"].append(neglog_prob.item())
		self.summary_dict["ldj"].append(neg_ldj.item())
		self.summary_dict["beta"] = self.beta_scheduler.get(iteration)
		self._ldj_per_layer_to_summary(ldj_per_layer)

		return loss


	def _eval_batch(self, batch, is_test=False):
		x_in, x_length, x_channel_mask = self._preprocess_batch(batch)
		if isinstance(self.model, LSTMModel):
			return self._eval_batch_rnn(x_in, x_length, x_channel_mask)
		elif isinstance(self.model, DFModel):
			return self._eval_batch_df(x_in, x_length, x_channel_mask)
		elif isinstance(self.model, CNFLanguageModeling):
			return self._eval_batch_flow(x_in, x_length, x_channel_mask, is_test=is_test)

	def _eval_batch_rnn(self, x_in, x_length, x_channel_mask, **kwargs):
		logprob, _ = self.model(x_in, reverse=False, length=x_length, channel_padding_mask=x_channel_mask)
		loss, _, _ = self._calc_loss(torch.zeros_like(logprob), -logprob, x_length)
		# print(loss.item())
		return loss

	def _eval_batch_df(self, x_in, x_length, x_channel_mask, **kwargs):
		vocab_dict = self.dataset_class.get_vocabulary()
		x = F.one_hot(x_in, num_classes = len(vocab_dict)).float()
		z = self.model(x, reverse=False)

		base_log_probs_sm = torch.nn.functional.log_softmax(self.model.base_log_probs, dim=-1).to(x_in.device)
		neglog_prob = -(z*base_log_probs_sm*x_channel_mask).sum(dim=[1,2])
		loss, _, _ = self._calc_loss(torch.zeros_like(neglog_prob), neglog_prob, x_length)
		return loss

	def _eval_batch_flow(self, x_in, x_length, x_channel_mask, is_test=False, **kwargs):
		z, ldj, ldj_per_layer = self.model(x_in, reverse=False, get_ldj_per_layer=True, 
										   length=x_length)
		neglog_prob = -(self.prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
		neg_ldj = -ldj
		loss, _, _ = self._calc_loss(neg_ldj, neglog_prob, x_length)
		return loss


	def _calc_loss(self, neg_ldj, neglog_prob, x_length):
		if self.length_prior is None:
			neg_ldj = (neg_ldj / x_length.float())
			neglog_prob = (neglog_prob / x_length.float())
			loss = neg_ldj + neglog_prob
		else:
			neg_ldj = (neg_ldj / (x_length+1).float())
			neglog_prob = (neglog_prob / (x_length+1).float())
			# Prior for timestep
			log_p_T = [self.length_prior[l]*1.0/(l+1) for l in x_length.detach().cpu().numpy()]
			log_p_T = torch.FloatTensor(log_p_T).to(get_device())
			loss = neg_ldj + neglog_prob + log_p_T

		loss = loss.mean()
		neg_ldj = neg_ldj.mean()
		neglog_prob = neglog_prob.mean()
		# print(neglog_prob.item())
		return loss, neg_ldj, neglog_prob


	def _preprocess_batch(self, batch):
		if isinstance(batch, tuple):
			x_in, x_length = batch
			if not isinstance(self.model, DFModel):
				x_in = x_in[:,:x_length.max()]
			x_channel_mask = create_channel_mask(x_length, max_len=x_in.size(1))
		else:
			x_in = batch
			x_length = x_in.new_zeros(x_in.size(0), dtype=torch.long) + x_in.size(1)
			x_channel_mask = x_in.new_ones(x_in.size(0), x_in.size(1), 1, dtype=torch.float32)
		return x_in, x_length, x_channel_mask


	def initialize(self, num_batches=1):
		if self.model.need_data_init():
			print("Preparing data dependent initialization...")
			batch_list = []
			for _ in range(num_batches):
				batch = self._get_next_batch()
				batch = TaskTemplate.batch_to_device(batch)
				x_in, x_length, _ = self._preprocess_batch(batch)
				batch_tuple = (x_in, {"length": x_length})
				batch_list.append(batch_tuple)
			self.model.initialize_data_dependent(batch_list)

class TrainTemplate:
	"""
	Template class to handle the training loop.
	Each experiment contains a experiment-specific training class inherting from this template class.
	"""

	def __init__(self, model_params, optimizer_params, batch_size, checkpoint_path, debug=False, name_prefix="", multi_gpu=False):
		self.batch_size = batch_size
		model_name = get_param_val(model_params, "model_name", default_val="CNF")
		self.name_prefix = name_prefix.strip()+model_name # Remove possible spaces. Name is used for creating default checkpoint path
		self.model_params = model_params
		self.optimizer_params = optimizer_params
		## Load model
		self.model = self._create_model(model_params)
		if multi_gpu: # Testing for multi-gpu if selected
			num_gpus = torch.cuda.device_count()
			if num_gpus == 0:
				print("[#] WARNING: Multi-GPU training failed because no GPU was detected. Continuing with single GPU...")
			elif num_gpus == 1:
				print("[#] WARNING: Multi-GPU training failed because only a single GPU is available. Continuing with single GPU...")
			else:
				print("Preparing to use %i GPUs..." % (num_gpus))
				self.model = WrappedDataParallel(self.model)

		self.model = self.model.to(get_device())
		## Load task
		self.task = self._create_task(model_params, debug=debug)
		## Load optimizer and checkpoints
		self._create_optimizer(model_params, optimizer_params)
		self._prepare_checkpoint(checkpoint_path)


	def _create_model(self, model_params):
		# To be implemented by the inherting class
		raise NotImplementedError


	def _create_task(self, model_params, debug=False):
		# To be implemented by the inherting class
		raise NotImplementedError


	def _create_optimizer(self, model_params, optimizer_params):
		parameters_to_optimize = self.model.parameters()
		self.optimizer = create_optimizer_from_args(parameters_to_optimize, optimizer_params, model_params, self.model)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, optimizer_params["lr_decay_step"], gamma=optimizer_params["lr_decay_factor"])
		self.lr_minimum = optimizer_params["lr_minimum"]


	def _prepare_checkpoint(self, checkpoint_path):
		if checkpoint_path is None:
			current_date = datetime.datetime.now()
			checkpoint_path = "checkpoints/%s%02d_%02d_%02d__%02d_%02d_%02d/" % ((self.name_prefix + "_") if len(self.name_prefix)>0 else "", current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)
		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)
		self.checkpoint_path = checkpoint_path


	def train_model(self, max_iterations=1e6, loss_freq=50, eval_freq=2000, save_freq=1e5, max_gradient_norm=0.25, no_model_checkpoints=False):

		parameters_to_optimize = self.model.parameters()

		# Setup dictionary to save evaluation details in
		checkpoint_dict = self.load_recent_model()
		start_iter = get_param_val(checkpoint_dict, "iteration", 0, warning_if_default=False) # Iteration to start from
		evaluation_dict = get_param_val(checkpoint_dict, "evaluation_dict", dict(), warning_if_default=False) # Dictionary containing validation performances over time
		best_save_dict = get_param_val(checkpoint_dict, "best_save_dict", {"file": None, "metric": 1e6, "detailed_metrics": None, "test": None}, warning_if_default=False) # 
		best_save_iter = best_save_dict["file"]
		last_save = None if start_iter == 0 else self.get_checkpoint_filename(start_iter)
		if last_save is not None and not os.path.isfile(last_save):
			print("[!] WARNING: Could not find last checkpoint file specified as " + last_save)
			last_save = None
		test_NLL = None # Possible test performance determined in the end of the training

		# Initialize tensorboard writer
		writer = SummaryWriter(self.checkpoint_path)

		# Function for saving model. Add here in the dictionary necessary parameters that should be saved
		def save_train_model(iteration, only_weights=True):
			if no_model_checkpoints:
				return
			checkpoint_dict = {
				"iteration": iteration,
				"best_save_dict": best_save_dict,
				"evaluation_dict": evaluation_dict
			}
			self.save_model(iteration, checkpoint_dict, save_optimizer=not only_weights)

		# Function to export the current results to a txt file
		def export_result_txt():
			if best_save_iter is not None:
				with open(os.path.join(self.checkpoint_path, "results.txt"), "w") as f:
					f.write("Best validation performance: %s\n" % (str(best_save_dict["metric"])))
					f.write("Best iteration: %i\n" % int(str(best_save_iter).split("_")[-1].split(".")[0]))
					f.write("Best checkpoint: %s\n" % str(best_save_iter))
					f.write("Detailed metrics\n")
					for metric_name, metric_val in best_save_dict["detailed_metrics"].items():
						f.write("-> %s: %s\n" % (metric_name, str(metric_val)))
					if "test" in best_save_dict and best_save_dict["test"] is not None:
						f.write("Test - Detailed metrics\n")
						for metric_name, metric_val in best_save_dict["test"].items():
							f.write("[TEST] -> %s: %s\n" % (metric_name, str(metric_val)))
					f.write("\n")
				
		# "Trackers" are moving averages. We use them to log the loss and time needed per training iteration
		time_per_step = Tracker()
		train_losses = Tracker()

		# Try-catch if user terminates
		try:
			index_iter = -1
			self.model.eval()
			self.task.initialize()
			print("="*50 + "\nStarting training...\n"+"="*50)
			self.model.train()

			print("Performing initial evaluation...")
			self.model.eval()
			eval_NLL, detailed_scores = self.task.eval(initial_eval=True)
			self.model.train()
			write_dict_to_tensorboard(writer, detailed_scores, base_name="eval", iteration=start_iter)		
			
			for index_iter in range(start_iter, int(max_iterations)):
				
				# Training step
				start_time = time.time()
				loss = self.task.train_step(iteration=index_iter)
				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(parameters_to_optimize, max_gradient_norm)
				if self.model.model_name in ["DAF", "DBF"]:
					torch.nn.utils.clip_grad_norm_(self.model.base_log_probs, max_gradient_norm)
				self.optimizer.step()
				if self.optimizer.param_groups[0]['lr'] > self.lr_minimum:
					self.lr_scheduler.step()
				end_time = time.time()

				time_per_step.add(end_time - start_time)
				train_losses.add(loss.item())

				# Statement for detecting NaN values 
				if torch.isnan(loss).item():
					print("[!] ERROR: Loss is NaN!" + str(loss.item()))
				for name, param in self.model.named_parameters():
					if param.requires_grad:
						if torch.isnan(param).sum() > 0:
							print("[!] ERROR: Parameter %s has %s NaN values!\n" % (name, str(torch.isnan(param).sum())) + \
								  "Grad values NaN: %s.\n" % (str(torch.isnan(param.grad).sum()) if param.grad is not None else "no gradients") + \
								  "Grad values avg: %s.\n" % (str(param.grad.abs().mean()) if param.grad is not None else "no gradients") + \
								  "Last loss: %s" % (str(loss)))

				# Printing current loss etc. for debugging
				if (index_iter + 1) % loss_freq == 0:
					loss_avg = train_losses.get_mean(reset=True)
					bpd_avg = self.task.loss_to_bpd(loss_avg)
					train_time_avg = time_per_step.get_mean(reset=True)
					max_memory = torch.cuda.max_memory_allocated(device=get_device())/1.0e9 if torch.cuda.is_available() else -1
					print("Training iteration %i|%i (%4.2fs). Loss: %6.5f, Bpd: %6.4f [Mem: %4.2fGB]" % (index_iter+1, max_iterations, train_time_avg, loss_avg, bpd_avg, max_memory))
					writer.add_scalar("train/loss", loss_avg, index_iter + 1)
					writer.add_scalar("train/bpd", bpd_avg, index_iter + 1)
					writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], index_iter+1)
					writer.add_scalar("train/training_time", train_time_avg, index_iter+1)

					self.task.add_summary(writer, index_iter + 1, checkpoint_path=self.checkpoint_path)
	

				# Performing evaluation every "eval_freq" steps
				if (index_iter + 1) % eval_freq == 0:
					self.model.eval()
					eval_NLL, detailed_scores = self.task.eval()
					self.model.train()

					write_dict_to_tensorboard(writer, detailed_scores, base_name="eval", iteration=index_iter+1)

					# If model performed better on validation than any other iteration so far => save it and eventually replace old model
					if eval_NLL < best_save_dict["metric"]:
						best_save_iter = self.get_checkpoint_filename(index_iter+1)
						best_save_dict["metric"] = eval_NLL
						best_save_dict["detailed_metrics"] = detailed_scores
						if not os.path.isfile(best_save_iter):
							print("Saving model at iteration " + str(index_iter+1))
							if best_save_dict["file"] is not None and os.path.isfile(best_save_dict["file"]):
								print("Removing checkpoint %s..." % best_save_dict["file"])
								os.remove(best_save_dict["file"])
							if last_save is not None and os.path.isfile(last_save):
								print("Removing checkpoint %s..." % last_save)
								os.remove(last_save)
							best_save_dict["file"] = best_save_iter
							last_save = best_save_iter
							save_train_model(index_iter+1)
						self.task.export_best_results(self.checkpoint_path, index_iter + 1)
						export_result_txt()
					evaluation_dict[index_iter + 1] = best_save_dict["metric"]

				# Independent of evaluation, the model is saved every "save_freq" steps. This prevents loss of information if model does not improve for a while
				if (index_iter + 1) % save_freq == 0 and not os.path.isfile(self.get_checkpoint_filename(index_iter+1)):
					save_train_model(index_iter + 1)
					if last_save is not None and os.path.isfile(last_save) and last_save != best_save_iter:
						print("Removing checkpoint %s..." % last_save)
						os.remove(last_save)
					last_save = self.get_checkpoint_filename(index_iter+1)
			## End training loop
			
			# Before testing, load best model and check whether its validation performance is in the right range (to prevent major loading issues)
			if not no_model_checkpoints and best_save_iter is not None:
				load_model(best_save_iter, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
				eval_NLL, detailed_scores = self.task.eval()
				if eval_NLL != best_save_dict["metric"]:
					if abs(eval_NLL - best_save_dict["metric"]) > 1e-1:
						print("[!] WARNING: new evaluation significantly differs from saved one (%s vs %s)! Probably a mistake in the saving/loading part..." % (str(eval_NLL), str(best_save_dict["metric"])))
					else:
						print("[!] WARNING: new evaluation sligthly differs from saved one (%s vs %s)." % (str(eval_NLL), str(best_save_dict["metric"])))
			else:
				print("Using last model as no models were saved...")
			
			# Testing the trained model
			test_NLL, detailed_scores = self.task.test()
			print("="*50+"\nTest performance: %lf" % (test_NLL))
			detailed_scores["original_NLL"] = test_NLL
			best_save_dict["test"] = detailed_scores
			self.task.finalize_summary(writer, max_iterations, self.checkpoint_path)

		# If user terminates training early, replace last model saved per "save_freq" steps by current one
		except KeyboardInterrupt:
			if index_iter > 0:
				print("User keyboard interrupt detected. Saving model at step %i..." % (index_iter))
				save_train_model(index_iter + 1)
			else:
				print("User keyboard interrupt detected before starting to train.")
			if last_save is not None and os.path.isfile(last_save) and not any([val == last_save for _, val in best_save_dict.items()]):
				os.remove(last_save)

		export_result_txt()

		writer.close()


	def get_checkpoint_filename(self, iteration):
		checkpoint_file = os.path.join(self.checkpoint_path, 'checkpoint_' + str(iteration).zfill(7) + ".tar")
		return checkpoint_file


	def save_model(self, iteration, add_param_dict, save_embeddings=False, save_optimizer=True):
		checkpoint_file = self.get_checkpoint_filename(iteration)
		if isinstance(self.model, nn.DataParallel):
			model_dict = self.model.module.state_dict()
		else:
			model_dict = self.model.state_dict()
		
		checkpoint_dict = {
			'model_state_dict': model_dict
		}
		if save_optimizer:
			checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
			checkpoint_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()
		checkpoint_dict.update(add_param_dict)
		torch.save(checkpoint_dict, checkpoint_file)


	def load_recent_model(self):
		checkpoint_dict = load_model(self.checkpoint_path, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
		return checkpoint_dict


	def evaluate_model(self, checkpoint_model=None):
		## Function for evaluation/testing of a model

		# Load the "best" model by first loading the most recent one and determining the "best" model
		checkpoint_dict = self.load_recent_model()
		best_save_dict = get_param_val(checkpoint_dict, "best_save_dict", {"file": None, "metric": -1, "detailed_metrics": dict()}, warning_if_default=True) # 
		best_save_iter = best_save_dict["file"]
		if not os.path.isfile(best_save_iter):
			splits = best_save_iter.split("/")
			checkpoint_index = splits.index("checkpoints")
			best_save_iter = "/".join(splits[checkpoint_index:])
		if not os.path.isfile(best_save_iter):
			print("[!] WARNING: Tried to load best model \"%s\", but file does not exist" % (best_save_iter))
		else:
			load_model(best_save_iter, model=self.model)

		# Print saved information of performance on validation set
		print("\n" + "-"*100 + "\n")
		print("Best evaluation iteration", best_save_iter)
		print("Best evaluation metric", best_save_dict["metric"])
		print("Detailed metrics")
		for metric_name, metric_val in best_save_dict["detailed_metrics"].items():
			print("-> %s: %s" % (metric_name, str(metric_val)))
		print("\n" + "-"*100 + "\n")

		# Test model
		self.task.checkpoint_path = self.checkpoint_path
		eval_metric, detailed_metrics = self.task.test()

		# Print test results
		out_dict = {}
		print("Evaluation metric", eval_metric)
		print("Detailed metrics")
		for metric_name, metric_val in detailed_metrics.items():
			print("-> %s: %s" % (metric_name, str(metric_val)))
			out_dict[metric_name] = str(metric_val) if isinstance(metric_val, torch.Tensor) else metric_val
		print("\n" + "-"*100 + "\n")

		# Save test results externally
		with open(os.path.join(self.checkpoint_path, "eval_metrics.json"), "w") as f: 
			json.dump(out_dict, f, indent=4)

class TrainLanguageModeling(TrainTemplate):
	
	def __init__(self, model_params, optimizer_params, batch_size, checkpoint_path, debug=False, **kwargs):
		super().__init__(model_params, optimizer_params, batch_size, checkpoint_path, debug=debug, name_prefix="LanguageModeling_", **kwargs)

	def _create_model(self, model_params):
		dataset_name = get_param_val(self.model_params, "dataset", default_val="penntreebank")
		dataset_class = TaskLanguageModeling.get_dataset_class(dataset_name)
		vocab_dict = dataset_class.get_vocabulary()
		vocab_torchtext = dataset_class.get_torchtext_vocab()

		model_name = get_param_val(self.model_params, "model_name", default_val="CNF")
		if model_name == "RNN":
			model = LSTMModel(num_classes=len(vocab_dict), vocab=vocab_torchtext, model_params=model_params)
		elif model_name == "CNF":
			model = CNFLanguageModeling(model_params=model_params, vocab_size=len(vocab_dict), vocab=vocab_torchtext, dataset_class=dataset_class)
		elif model_name in ["DAF", "DBF"]:
			model = DFModel(num_classes=len(vocab_dict), batch_size=self.batch_size, model_params=model_params, model_name=model_name)
		return model

	def _create_task(self, model_params, debug=False):
		task = TaskLanguageModeling(self.model, model_params, debug=debug, batch_size=self.batch_size)
		return task