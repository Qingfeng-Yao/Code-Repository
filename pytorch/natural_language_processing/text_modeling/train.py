import argparse
import sys
import os
from glob import glob
import shutil
import pickle

import torch

from distributions import PriorDistribution
from utils import set_debug_level, load_args, args_to_params, PARAM_CONFIG_FILE
from models import TrainLanguageModeling

def start_training(args):
	"""
	Function to start a training loop.
	"""
	if args.cluster:
		set_debug_level(2)
		loss_freq = 250
	else:
		set_debug_level(0)
		loss_freq = 2
		if args.debug:
			# To find possible errors easier, activate anomaly detection. Note that this slows down training
			torch.autograd.set_detect_anomaly(True) 

	if args.print_freq > 0:
		loss_freq = args.print_freq

	only_eval = args.only_eval

	if args.load_config:
		if args.checkpoint_path is None:
			print("[!] ERROR: Please specify the checkpoint path to load the config from.")
			sys.exit(1)
		debug = args.debug
		checkpoint_path = args.checkpoint_path
		args = load_args(args.checkpoint_path)
		args.clean_up = False
		args.checkpoint_path = checkpoint_path
		if only_eval:
			args.use_multi_gpu = False
			args.debug = debug

	# Setup training
	model_params, optimizer_params = args_to_params(args) # make params to dict, set seed, model_params include prior distribution, cate encoding, scheduler
	trainModule = TrainLanguageModeling(model_params=model_params,
								optimizer_params=optimizer_params, 
								batch_size=args.batch_size,
								checkpoint_path=args.checkpoint_path, 
								debug=args.debug,
								multi_gpu=args.use_multi_gpu
								)

	# Function for cleaning up the checkpoint directory
	def clean_up_dir():
		assert str(trainModule.checkpoint_path) not in ["/", "/home/", "/lhome/"], \
				"[!] ERROR: Checkpoint path is \"%s\" and is selected to be cleaned. This is probably not wanted..." % str(trainModule.checkpoint_path)
		print("Cleaning up directory " + str(trainModule.checkpoint_path) + "...")
		for file_in_dir in sorted(glob(os.path.join(trainModule.checkpoint_path, "*"))):
			print("Removing file " + file_in_dir)
			try:
				if os.path.isfile(file_in_dir):
					os.remove(file_in_dir)
				elif os.path.isdir(file_in_dir): 
					shutil.rmtree(file_in_dir)
			except Exception as e:
				print(e)

	if args.restart and args.checkpoint_path is not None and os.path.isdir(args.checkpoint_path) and not only_eval:
		clean_up_dir()

	if not only_eval:
		# Save argument namespace object for loading/evaluation
		args_filename = os.path.join(trainModule.checkpoint_path, PARAM_CONFIG_FILE)
		with open(args_filename, "wb") as f:
			pickle.dump(args, f)

		# Start training
		trainModule.train_model(args.max_iterations, loss_freq=loss_freq, eval_freq=args.eval_freq, save_freq=args.save_freq, no_model_checkpoints=args.no_model_checkpoints)

		# Cleaning up the checkpoint directory afterwards if selected
		if args.clean_up:
			clean_up_dir()
			os.rmdir(trainModule.checkpoint_path)
	else:
		# Only evaluating the model. Should be combined with loading a model.
		# However, the recommended way of evaluating a model is by the "eval.py" file in the experiment folder(s).
		trainModule.evaluate_model()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# model and dataset param; cuda is set in utils.py
	parser.add_argument("--model_name", help="Name of model. Options: RNN(LSTM), CNF(Categorical Normalizing Flow)", type=str, default="CNF")
	parser.add_argument("--dataset", help="Name of the dataset to train on. Options: penntreebank, text8, wikitext", type=str, default="penntreebank")
	parser.add_argument("--batch_size", help="Batch size used during training", type=int, default=64)
	parser.add_argument("--max_seq_len", help="Maximum sequence length of training sentences.", type=int, default=256)
	
	# Training parameters
	parser.add_argument("--seed", help="Seed to make experiments reproducable", type=int, default=42)
	parser.add_argument("--max_iterations", help="Maximum number of epochs to train.", type=int, default=1e6)
	parser.add_argument("--print_freq", help="In which frequency loss information should be printed. Default: 250 if args.cluster, else 2", type=int, default=-1)
	parser.add_argument("--eval_freq", help="In which frequency the model should be evaluated (in number of iterations). Default: 2000", type=int, default=2000)
	parser.add_argument("--save_freq", help="In which frequency the model should be saved (in number of iterations). Default: 10,000", type=int, default=1e4)
	parser.add_argument("--use_multi_gpu", help="Whether to use all GPUs available or only one.", action="store_true")
	# Arguments for loading and saving models.
	parser.add_argument("--restart", help="Does not load old checkpoints, and deletes those if checkpoint path is specified (including tensorboard file etc.)", action="store_true")
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints should be saved", type=str, default=None)
	parser.add_argument("--load_config", help="Tries to find parameter file in checkpoint path, and loads all given parameters from there", action="store_true")
	parser.add_argument("--no_model_checkpoints", help="If selected, no model checkpoints will be saved", action="store_true")
	parser.add_argument("--only_eval", help="If selected, no training is performed but only an evaluation will be executed.", action="store_true")
	# Controlling the output
	parser.add_argument("--cluster", help="Enable option if code is executed on cluster. Reduces output size", action="store_true")
	parser.add_argument("-d", "--debug", help="Whether debug output should be activated or not", action="store_true")
	parser.add_argument("--clean_up", help="Whether to remove all files after finishing or not", action="store_true")
	# Optimizer parameters
	parser.add_argument("--learning_rate", help="Learning rate of the optimizer", type=float, default=7.5e-4)
	parser.add_argument("--lr_decay_factor", help="Decay of learning rate of the optimizer, applied after \"lr_decay_step\" training iterations.", type=float, default=0.999975)
	parser.add_argument("--lr_decay_step", help="Number of steps after which learning rate should be decreased", type=float, default=1)
	parser.add_argument("--lr_minimum", help="Minimum learning rate that should be scheduled. Default: no limit.", type=float, default=0.0)
	parser.add_argument("--weight_decay", help="Weight decay of the optimizer", type=float, default=0.0)
	parser.add_argument("--optimizer", help="Which optimizer to use. 0: SGD, 1: Adam, 2: Adamax, 3: RMSProp, 4: RAdam, 5: Adam Warmup", type=int, default=4)
	parser.add_argument("--momentum", help="Apply momentum to SGD optimizer", type=float, default=0.0)
	parser.add_argument("--beta1", help="Value for beta 1 parameter in Adam-like optimizers", type=float, default=0.9)
	parser.add_argument("--beta2", help="Value for beta 2 parameter in Adam-like optimizers", type=float, default=0.999)
	parser.add_argument("--warmup", help="If Adam with Warmup is selected, this value determines the number of warmup iterations to use.", type=int, default=2000)
	
	# Add parameters for prior distribution
	add_name = "" # need in defining param dict
	parser.add_argument("--%sprior_dist_type" % add_name, help="Selecting the prior distribution that should be used. Options are: " + PriorDistribution.get_string_of_distributions(), type=int, default=PriorDistribution.LOGISTIC)
	parser.add_argument("--%sprior_dist_mu" % add_name, help="Center location of the distribution.", type=float, default=None)
	parser.add_argument("--%sprior_dist_sigma" % add_name, help="Scaling of the distribution.", type=float, default=None)
	parser.add_argument("--%sprior_dist_start_x" % add_name, help="If distribution is bounded, but should be shifted, this parameter determines the start position.", type=float, default=None)
	parser.add_argument("--%sprior_dist_stop_x" % add_name, help="If distribution is bounded, but should be shifted, this parameter determines the end position.", type=float, default=None)

	# Add parameters for categorical encoding
	postfix="" # need in defining param dict
	# General parameters
	parser.add_argument("--encoding_dim" + postfix, help="Dimensionality of the embeddings.", type=int, default=4)
	parser.add_argument("--encoding_dequantization" + postfix, help="If selected, variational dequantization is used for encoding categorical data.", action="store_true")
	parser.add_argument("--encoding_variational" + postfix, help="If selected, the encoder distribution is joint over categorical variables.", action="store_true")
	
	# Flow parameters
	parser.add_argument("--encoding_num_flows" + postfix, help="Number of flows used in the embedding layer.", type=int, default=0)
	parser.add_argument("--encoding_hidden_layers" + postfix, help="Number of hidden layers of flows used in the parallel embedding layer.", type=int, default=2)
	parser.add_argument("--encoding_hidden_size" + postfix, help="Hidden size of flows used in the parallel embedding layer.", type=int, default=128)
	parser.add_argument("--encoding_num_mixtures" + postfix, help="Number of mixtures used in the coupling layers (if applicable).", type=int, default=8)
	
	# Decoder parameters
	parser.add_argument("--encoding_use_decoder" + postfix, help="If selected, we use a decoder instead of calculating the likelihood by inverting all flows.", action="store_true")
	parser.add_argument("--encoding_dec_num_layers" + postfix, help="Number of hidden layers used in the decoder of the parallel embedding layer.", type=int, default=1)
	parser.add_argument("--encoding_dec_hidden_size" + postfix, help="Hidden size used in the decoder of the parallel embedding layer.", type=int, default=64)
	
	# Coupling layer parameters
	parser.add_argument("--coupling_hidden_size", help="Hidden size of the coupling layers.", type=int, default=1024)
	parser.add_argument("--coupling_hidden_layers", help="Number of hidden layers in the coupling layers.", type=int, default=2)
	parser.add_argument("--coupling_num_flows", help="Number of coupling layers to use.", type=int, default=1)
	parser.add_argument("--coupling_num_mixtures", help="Number of mixtures used in the coupling layers.", type=int, default=64)
	parser.add_argument("--coupling_dropout", help="Dropout to use in the networks.", type=float, default=0.0)
	parser.add_argument("--coupling_input_dropout", help="Input dropout rate to use in the networks.", type=float, default=0.0)
	
	# Parameter for schedulers
	default_vals = {"beta": ("exponential", 1.0, 2.0, 5000, 2, 0)}
	param_names = ["beta"] # need in defining param dict
	for name in param_names:
		default_type, default_end_val, default_start_val, default_step_size, default_logit, default_delay = "constant", 0.0, 0.0, 100, 2.0, 0
		if default_vals is not None and name in default_vals:
			default_type, default_end_val, default_start_val, default_step_size, default_logit, default_delay = default_vals[name]
		parser.add_argument("--%s_scheduler_type" % name, help="Which type of scheduler to use for the parameter %s. Options: constant, sigmoid, exponential" % name, type=str, default=default_type)
		parser.add_argument("--%s_scheduler_end_val" % name, help="Value of the parameter %s which should be reached for t->infinity." % name, type=float, default=default_end_val)
		parser.add_argument("--%s_scheduler_start_val" % name, help="Value of the parameter %s at t=0" % name, type=float, default=default_start_val)
		parser.add_argument("--%s_scheduler_step_size" % name, help="Step size which should be used in the scheduler for %s." % name, type=int, default=default_step_size)
		parser.add_argument("--%s_scheduler_logit" % name, help="Logit which should be used in the scheduler for %s." % name, type=float, default=default_logit)
		parser.add_argument("--%s_scheduler_delay" % name, help="Delay which should be used in the scheduler for %s." % name, type=int, default=default_delay)

	# Parse given parameters and start training
	args = parser.parse_args()
	start_training(args)