import argparse
import numpy as np
import sys
import datetime
import os
from glob import glob
import time
import shutil
import random
import pickle
from statistics import mean

import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from datasets import PennTreeBankDataset, Text8Dataset, WikiTextDataset
from categorical_flows import *
import categorical_util

## 参数设置
parser = argparse.ArgumentParser(description='pytorch language modeling using categorical normalizing flows')
parser.add_argument(
    '--dataset',
    type=str, 
    default='penntreebank',
    help='name of the dataset to train on, options: penntreebank, text8, wikitext')
parser.add_argument(
    '--cuda-device',
    type=str, 
    default='cuda:0',
    help='cuda:0 | ...')
parser.add_argument(
    '--use_rnn',
    action='store_true',
    default=False,
    help='if selected, a RNN is used as model instead of a Categorical Normalizing Flow')

parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disable cuda training')
parser.add_argument(
    '--seed', type=int, default=42, help='random seed')
parser.add_argument(
    '--checkpoint_path',
    type=str, 
    default=None,
    help='folder(name) where checkpoints should be saved')
parser.add_argument(
    '--clean_up',
    action='store_false',
    default=True,
    help='whether to remove all files after finishing or not')
parser.add_argument(
    '--print_freq', type=int, default=250, help='in which frequency loss information should be printed. Default: 250 or 2')
parser.add_argument(
    '--eval_freq', type=int, default=2000, help='in which frequency the model should be evaluated (in number of iterations). Default: 2000')
parser.add_argument(
    '--save_freq', type=int, default=1e4, help='in which frequency the model should be saved (in number of iterations). Default: 10,000')

parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='input batch size for training')
parser.add_argument(
    '--max_seq_len',
    type=int,
    default=256,
    help='maximum sequence length of training sentences')

parser.add_argument(
    '--learning_rate', type=float, default=7.5e-4, help='learning rate')
parser.add_argument(
    '--lr_decay_factor', type=float, default=0.999975, help='decay of learning rate of the optimizer, applied after \"lr_decay_step\" training iterations')
parser.add_argument(
    '--lr_decay_step', type=float, default=1, help='number of steps after which learning rate should be decreased')
parser.add_argument(
    '--lr_minimum', type=float, default=0.0, help='minimum learning rate that should be scheduled, default: no limit')
parser.add_argument(
    '--weight_decay', type=float, default=0.0, help='weight decay applied to all weights')
parser.add_argument(
    '--optimizer', type=int,  default=4, help='which optimizer to use. 0: SGD, 1: Adam, 2: Adamax, 3: RMSProp, 4: RAdam, 5: Adam Warmup')
parser.add_argument(
    '--momentum', type=float, default=0.0, help='apply momentum to SGD optimizer')
parser.add_argument(
    '--beta1', type=float, default=0.9, help='value for beta 1 parameter in Adam-like optimizers')
parser.add_argument(
    '--beta2', type=float, default=0.999, help='value for beta 2 parameter in Adam-like optimizers')
parser.add_argument(
    '--warmup', type=int, default=2000, help='if Adam with Warmup is selected, this value determines the number of warmup iterations to use')
parser.add_argument(
    '--beta_scheduler_type', type=str, default='exponential', help='which type of scheduler to use for the parameter beta, options: constant, sigmoid, exponential')
parser.add_argument(
    '--beta_scheduler_end_val', type=float, default=1.0, help='value of the parameter beta which should be reached for t->infinity')
parser.add_argument(
    '--beta_scheduler_start_val', type=float, default=2.0, help='value of the parameter beta at t=0')
parser.add_argument(
    '--beta_scheduler_step_size', type=int, default=5000, help='step size which should be used in the scheduler for beta')
parser.add_argument(
    '--beta_scheduler_logit', type=float, default=2, help='logit which should be used in the scheduler for beta')
parser.add_argument(
    '--beta_scheduler_delay', type=int, default=0, help='delay which should be used in the scheduler for beta')
parser.add_argument(
    '--max_iterations', type=int, default=1e6, help='maximum number of epochs to train')

parser.add_argument(
    '--prior_dist_type', type=int, default=1, help='selecting the prior distribution that should be used, options are: GAUSSIAN: 0;LOGISTIC: 1')
parser.add_argument(
    '--prior_dist_mu', type=float, default=0.0, help='center location of the distribution')
parser.add_argument(
    '--prior_dist_sigma', type=float, default=1.0, help='scaling of the distribution')
parser.add_argument(
    '--prior_dist_start_x', type=float, default=None, help='if distribution is bounded, but should be shifted, this parameter determines the start position')
parser.add_argument(
    '--prior_dist_stop_x', type=float, default=None, help='if distribution is bounded, but should be shifted, this parameter determines the end position')

parser.add_argument(
    '--encoding_dim', type=int, default=4, help='dimensionality of the embeddings')
parser.add_argument(
    '--encoding_dequantization',
    action='store_true',
    default=False,
    help='if selected, variational dequantization is used for encoding categorical data')
parser.add_argument(
    '--encoding_variational',
    action='store_true',
    default=False,
    help='if selected, the encoder distribution is joint over categorical variables')
parser.add_argument(
    '--encoding_num_flows', type=int, default=0, help='number of flows used in the embedding layer')
parser.add_argument(
    '--encoding_hidden_layers', type=int, default=2, help='number of hidden layers of flows used in the parallel embedding layer')
parser.add_argument(
    '--encoding_hidden_size', type=int, default=128, help='hidden size of flows used in the parallel embedding layer')
parser.add_argument(
    '--encoding_num_mixtures', type=int, default=8, help='number of mixtures used in the coupling layers (if applicable)')

parser.add_argument(
    '--encoding_use_decoder',
    action='store_true',
    default=False,
    help='if selected, we use a decoder instead of calculating the likelihood by inverting all flows')
parser.add_argument(
    '--encoding_dec_num_layers', type=int, default=1, help='number of hidden layers used in the decoder of the parallel embedding layer')
parser.add_argument(
    '--encoding_dec_hidden_size', type=int, default=64, help='hidden size used in the decoder of the parallel embedding layer')

parser.add_argument(
    '--coupling_hidden_size', type=int, default=1024, help='hidden size of the coupling layers')
parser.add_argument(
    '--coupling_hidden_layers', type=int, default=2, help='number of hidden layers in the coupling layers')
parser.add_argument(
    '--coupling_num_flows', type=int, default=1, help='number of coupling layers to use')
parser.add_argument(
    '--coupling_num_mixtures', type=int, default=64, help='number of mixtures used in the coupling layers')
parser.add_argument(
    '--coupling_dropout', type=float, default=0.0, help='dropout to use in the networks')
parser.add_argument(
    '--coupling_input_dropout', type=float, default=0.0, help='input dropout rate to use in the networks')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.cuda_device if args.cuda else "cpu")

loss_freq = args.print_freq

# Set seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available: 
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_params, optimizer_params = categorical_util.args_to_params(args)

## 数据下载
dataset_name = categorical_util.get_param_val(model_params, "dataset", default_val="penntreebank")

if dataset_name == "penntreebank":
    dataset_class = PennTreeBankDataset
elif dataset_name == "text8":
    dataset_class = Text8Dataset
elif dataset_name == "wikitext":
    dataset_class = WikiTextDataset
else:
    assert False, "[!] ERROR: Unknown dataset class \"%s\"" % (dataset_name)

vocab_dict = dataset_class.get_vocabulary()
vocab_torchtext = dataset_class.get_torchtext_vocab()

train_epoch = 0

print("Loading dataset %s..." % args.dataset)
train_dataset = dataset_class(max_seq_len=args.max_seq_len, train=True)
val_dataset = dataset_class(max_seq_len=args.max_seq_len, val=True)
test_dataset = dataset_class(max_seq_len=args.max_seq_len, test=True)

num_workers = 1
drop_last = False
def _init_fn(worker_id):
    np.random.seed(42)
# Initializes all data loaders with the loaded datasets
if hasattr(train_dataset, "get_sampler"):
    train_data_loader = data.DataLoader(train_dataset, batch_sampler=train_dataset.get_sampler(args.batch_size, drop_last=drop_last), pin_memory=True, 
                                                num_workers=num_workers, worker_init_fn=_init_fn)
    val_data_loader = data.DataLoader(val_dataset, batch_sampler=val_dataset.get_sampler(args.batch_size, drop_last=drop_last), pin_memory=True, num_workers=num_workers, worker_init_fn=_init_fn)
    test_data_loader = data.DataLoader(test_dataset, batch_sampler=test_dataset.get_sampler(args.batch_size, drop_last=drop_last), pin_memory=True, num_workers=num_workers, worker_init_fn=_init_fn)		
else:
    train_data_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=drop_last, num_workers=num_workers,
                                                worker_init_fn=_init_fn)
    val_data_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=drop_last, num_workers=num_workers, worker_init_fn=_init_fn)
    test_data_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=drop_last, num_workers=num_workers, worker_init_fn=_init_fn)
train_data_loader_iter = iter(train_data_loader) # Needed to retrieve batch by batch from dataset

## 模型及优化器
use_rnn = categorical_util.get_param_val(model_params, "use_rnn", default_val=False)
if use_rnn:
    model = LSTMModel(num_classes=len(vocab_dict), vocab=vocab_torchtext, model_params=model_params)
# else:
#     model = FlowLanguageModeling(args, vocab_size=len(vocab_dict), vocab=vocab_torchtext, dataset_class=dataset_class)

# print(model)
model = model.to(device)

### 创建先验分布
prior_dist_params = categorical_util.get_param_val(model_params, "prior_distribution", allow_default=False, error_location="prior_distribution - create")
prior_distribution = categorical_util.create_prior_distribution(prior_dist_params)
### 创建beta scheduler
beta_scheduler = categorical_util.create_scheduler(model_params["beta"], "beta")

summary_dict = {"log_prob": list(), "ldj": list(), "z": list(), "beta": 0}

parameters_to_optimize = model.parameters()
optimizer = categorical_util.create_optimizer_from_args(parameters_to_optimize, optimizer_params)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_factor)
lr_minimum = args.lr_minimum

## 训练及测试
checkpoint_path = args.checkpoint_path
if checkpoint_path is None:
    current_date = datetime.datetime.now()
    checkpoint_path = "checkpoints/%s%02d_%02d_%02d__%02d_%02d_%02d/" % (("categorical_flows" + "_"), current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Save argument namespace object for loading/evaluation
PARAM_CONFIG_FILE = "param_config.pik"
args_filename = os.path.join(checkpoint_path, PARAM_CONFIG_FILE)
with open(args_filename, "wb") as f:
    pickle.dump(args, f)

# Start training
def get_checkpoint_filename(iteration):
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint_' + str(iteration).zfill(7) + ".tar")
    return checkpoint_file

checkpoint_dict = categorical_util.load_model(checkpoint_path, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
start_iter = categorical_util.get_param_val(checkpoint_dict, "iteration", 0, warning_if_default=False) # Iteration to start from
evaluation_dict = categorical_util.get_param_val(checkpoint_dict, "evaluation_dict", dict(), warning_if_default=False) # Dictionary containing validation performances over time
best_save_dict = categorical_util.get_param_val(checkpoint_dict, "best_save_dict", {"file": None, "metric": 1e6, "detailed_metrics": None, "test": None}, warning_if_default=False) 
best_save_iter = best_save_dict["file"]
last_save = None if start_iter == 0 else get_checkpoint_filename(start_iter)
if last_save is not None and not os.path.isfile(last_save):
    print("[!] WARNING: Could not find last checkpoint file specified as " + last_save)
    last_save = None
test_NLL = None # Possible test performance determined in the end of the training

# Initialize tensorboard writer
writer = SummaryWriter(checkpoint_path)

# Function for saving model. Add here in the dictionary necessary parameters that should be saved
def save_train_model(iteration, only_weights=True):
    checkpoint_dict = {
        "iteration": iteration,
        "best_save_dict": best_save_dict,
        "evaluation_dict": evaluation_dict
    }
	
    save_model(iteration, checkpoint_dict, save_optimizer=not only_weights)

def save_model(iteration, add_param_dict, save_embeddings=False, save_optimizer=True):
    checkpoint_file = get_checkpoint_filename(iteration)
    model_dict = model.state_dict()
    
    checkpoint_dict = {
        'model_state_dict': model_dict
    }
    if save_optimizer:
        checkpoint_dict['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint_dict['scheduler_state_dict'] = lr_scheduler.state_dict()
    checkpoint_dict.update(add_param_dict)
    torch.save(checkpoint_dict, checkpoint_file)

def export_result_txt():
    if best_save_iter is not None:
        with open(os.path.join(checkpoint_path, "results.txt"), "w") as f:
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
time_per_step = categorical_util.Tracker()
train_losses = categorical_util.Tracker()

def _get_next_batch():
    # Try to get next batch. If one epoch is over, the iterator throws an error, and we start a new iterator
    global train_data_loader_iter
    global train_epoch
    try:
        batch = next(train_data_loader_iter)
    except StopIteration:
        train_data_loader_iter = iter(train_data_loader)
        batch = next(train_data_loader_iter)
        train_epoch += 1
    return batch

def batch_to_device(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = tuple([b.to(device) for b in batch])
    else:
        batch = batch.to(device)
    return batch

def _preprocess_batch(batch):
    if isinstance(batch, tuple):
        x_in, x_length = batch
        x_in = x_in[:,:x_length.max()]
        x_channel_mask = categorical_util.create_channel_mask(x_length, max_len=x_in.size(1))
    else:
        x_in = batch
        x_length = x_in.new_zeros(x_in.size(0), dtype=torch.long) + x_in.size(1)
        x_channel_mask = x_in.new_ones(x_in.size(0), x_in.size(1), 1, dtype=torch.float32)
    return x_in, x_length, x_channel_mask

index_iter = -1
model.eval()
if model.need_data_init():
    print("Preparing data dependent initialization...")
    batch_list = []
    for _ in range(1):
        batch = _get_next_batch()
        batch = batch_to_device(batch)
        x_in, x_length, _ = _preprocess_batch(batch)
        batch_tuple = (x_in, {"length": x_length})
        batch_list.append(batch_tuple)
    model.initialize_data_dependent(batch_list)
print("="*50 + "\nStarting training...\n"+"="*50)
model.train()

def _eval_batch(batch, is_test=False):
    x_in, x_length, x_channel_mask = _preprocess_batch(batch)
    if isinstance(model, LSTMModel):
        return _eval_batch_rnn(x_in, x_length, x_channel_mask)
    else:
        return _eval_batch_flow(x_in, x_length, x_channel_mask, is_test=is_test)

def _eval_batch_rnn(x_in, x_length, x_channel_mask, **kwargs):
    logprob, _ = model(x_in, reverse=False, length=x_length, channel_padding_mask=x_channel_mask)
    loss = -logprob.mean()
    return loss

# def _eval_batch_flow(x_in, x_length, x_channel_mask, is_test=False, **kwargs):
#     z, ldj, ldj_per_layer = model(x_in, reverse=False, get_ldj_per_layer=True, 
#                                         length=x_length)
#     neglog_prob = -(prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
#     neg_ldj = -ldj
#     loss, _, _ = _calc_loss(neg_ldj, neglog_prob, x_length)
#     return loss

def _calc_loss(neg_ldj, neglog_prob, x_length):
    neg_ldj = (neg_ldj / x_length.float())
    neglog_prob = (neglog_prob / x_length.float())
    loss = neg_ldj + neglog_prob

    loss = loss.mean()
    neg_ldj = neg_ldj.mean()
    neglog_prob = neglog_prob.mean()
    return loss, neg_ldj, neglog_prob

def loss_to_bpd(loss):
	return (np.log2(np.exp(1)) * loss)

def _eval_finalize_metrics(detailed_metrics, is_test=False, initial_eval=False):
    # This function is called after evaluation finished. 
    # Can be used to add final metrics to the dictionary, which is added to the tensorboard.
    pass

def eval(data_loader=None, **kwargs):
    # Default: if no dataset is specified, we use validation dataset
    if data_loader is None:
        assert val_data_loader is not None, "[!] ERROR: Validation dataset not loaded. Please load the dataset beforehand for evaluation."
        data_loader = val_data_loader
    is_test = (data_loader == test_data_loader)

    start_time = time.time()
    torch.cuda.empty_cache()
    model.eval()
    
    # Prepare metrics
    total_nll, embed_nll, nll_counter = 0, 0, 0

    # Evaluation loop
    with torch.no_grad():
        for batch_ind, batch in enumerate(data_loader):
            # Put batch on correct device
            batch = batch_to_device(batch)
            # Evaluate single batch
            batch_size = batch[0].size(0) if isinstance(batch, tuple) else batch.size(0)
            batch_nll = _eval_batch(batch, is_test=is_test)
            total_nll += batch_nll.item() * batch_size
            nll_counter += batch_size

    avg_nll = total_nll / max(1e-5, nll_counter)
    detailed_metrics = {
        "negative_log_likelihood": avg_nll,
        "bpd": loss_to_bpd(avg_nll), # Bits per dimension
    }

    with torch.no_grad():
        _eval_finalize_metrics(detailed_metrics, is_test=is_test, **kwargs)

    model.train()
    eval_time = int(time.time() - start_time)
    print("Finished %s with bpd of %4.3f (%imin %is)" % ("testing" if data_loader == test_data_loader else "evaluation", detailed_metrics["bpd"], eval_time/60, eval_time%60))
    torch.cuda.empty_cache()

    if "loss_metric" in detailed_metrics:
        loss_metric = detailed_metrics["loss_metric"]
    else:
        loss_metric = avg_nll
    
    return loss_metric, detailed_metrics

print("Performing initial evaluation...")
model.eval()
eval_NLL, detailed_scores = eval(initial_eval=True)
model.train()
categorical_util.write_dict_to_tensorboard(writer, detailed_scores, base_name="eval", iteration=start_iter)

def _ldj_per_layer_to_summary(ldj_per_layer, pre_phrase="ldj_layer_"):
    for layer_index, layer_ldj in enumerate(ldj_per_layer):
        if isinstance(layer_ldj, tuple) or isinstance(layer_ldj, list):
            for i in range(len(layer_ldj)):
                categorical_util.append_in_dict(summary_dict, "ldj_layer_%i_%i" % (layer_index, i), layer_ldj[i].detach().mean().item())
        elif isinstance(layer_ldj, dict):
            for key, ldj_val in layer_ldj.items():
                categorical_util.append_in_dict(summary_dict, "ldj_layer_%i_%s" % (layer_index, key), ldj_val.detach().mean().item())
        else:
            categorical_util.append_in_dict(summary_dict, "ldj_layer_%i" % layer_index, layer_ldj.detach().mean().item())

def train_step(iteration=0):
    # Check if training data was correctly loaded
    if train_data_loader_iter is None:
        print("[!] ERROR: Iterator of the training data loader was None. Additional parameters: " + \
                "train_data_loader was %sloaded, " % ("not " if train_data_loader is None else "") + \
                "train_dataset was %sloaded." % ("not " if train_dataset is None else ""))
    
    # Get batch and put it on correct device
    batch = _get_next_batch()
    batch = batch_to_device(batch)

    # Perform task-specific training step
    return _train_batch(batch, iteration=iteration)

def _train_batch(batch, iteration=0):
    x_in, x_length, x_channel_mask = _preprocess_batch(batch)
    if isinstance(model, LSTMModel):
        return _train_batch_rnn(x_in, x_length, x_channel_mask)
    else:
        return _train_batch_flow(x_in, x_length, x_channel_mask, iteration=iteration)

def _train_batch_rnn(x_in, x_length, x_channel_mask, **kwargs):
    logprob, details = model(x_in, reverse=False, length=x_length, channel_padding_mask=x_channel_mask)
    summary_dict["log_prob"].append(-logprob.mean().item())
    _ldj_per_layer_to_summary([details])
    loss = -logprob.mean()
    return loss

# def _train_batch_flow(x_in, x_length, x_channel_mask, iteration=0, **kwargs):
#     z, ldj, ldj_per_layer = model(x_in, reverse=False, get_ldj_per_layer=True, 
#                                         beta=self.beta_scheduler.get(iteration),
#                                         length=x_length)
#     neglog_prob = -(prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
#     neg_ldj = -ldj
    
#     loss, neg_ldj, neglog_prob = _calc_loss(neg_ldj, neglog_prob, x_length)

#     summary_dict["log_prob"].append(neglog_prob.item())
#     summary_dict["ldj"].append(neg_ldj.item())
#     summary_dict["beta"] = beta_scheduler.get(iteration)
#     _ldj_per_layer_to_summary(ldj_per_layer)

#     return loss

def add_summary(writer, iteration, checkpoint_path=None):
	# Adding metrics collected during training to the tensorboard
	# Function can/should be extended if needed
	for key, val in summary_dict.items():
		summary_key = "train_%s/%s" % ("cateflow", key)
		if not isinstance(val, list): # If it is not a list, it is assumably a single scalar
			writer.add_scalar(summary_key, val, iteration)
			summary_dict[key] = 0.0
		elif len(val) == 0: # Skip an empty list
			continue
		elif not isinstance(val[0], list): # For a list of scalars, report the mean
			writer.add_scalar(summary_key, mean(val), iteration)
			summary_dict[key] = list()
		else: # List of lists indicates a histogram
			val = [v for sublist in val for v in sublist]
			writer.add_histogram(summary_key, np.array(val), iteration)
			summary_dict[key] = list()

def export_best_results(checkpoint_path, iteration):
    # This function is called if the last evaluation has been the best so far.
    # Can be used to add some output to the checkpoint directory.
    pass

for index_iter in range(start_iter, int(args.max_iterations)):
    # Training step
    start_time = time.time()
    loss = train_step(iteration=index_iter)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters_to_optimize, 0.25)
    optimizer.step()
    if optimizer.param_groups[0]['lr'] > lr_minimum:
        lr_scheduler.step()
    end_time = time.time()

    time_per_step.add(end_time - start_time)
    train_losses.add(loss.item())

    # Statement for detecting NaN values 
    if torch.isnan(loss).item():
        print("[!] ERROR: Loss is NaN!" + str(loss.item()))
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).sum() > 0:
                print("[!] ERROR: Parameter %s has %s NaN values!\n" % (name, str(torch.isnan(param).sum())) + \
                        "Grad values NaN: %s.\n" % (str(torch.isnan(param.grad).sum()) if param.grad is not None else "no gradients") + \
                        "Grad values avg: %s.\n" % (str(param.grad.abs().mean()) if param.grad is not None else "no gradients") + \
                        "Last loss: %s" % (str(loss)))

    # Printing current loss etc. for debugging
    if (index_iter + 1) % loss_freq == 0:
        loss_avg = train_losses.get_mean(reset=True)
        bpd_avg = loss_to_bpd(loss_avg)
        train_time_avg = time_per_step.get_mean(reset=True)
        max_memory = torch.cuda.max_memory_allocated(device=device)/1.0e9 if torch.cuda.is_available() else -1
        print("Training iteration %i|%i (%4.2fs). Loss: %6.5f, Bpd: %6.4f [Mem: %4.2fGB]" % (index_iter+1, args.max_iterations, train_time_avg, loss_avg, bpd_avg, max_memory))
        writer.add_scalar("train/loss", loss_avg, index_iter + 1)
        writer.add_scalar("train/bpd", bpd_avg, index_iter + 1)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], index_iter+1)
        writer.add_scalar("train/training_time", train_time_avg, index_iter+1)

        add_summary(writer, index_iter + 1, checkpoint_path=checkpoint_path)

    # Performing evaluation every "eval_freq" steps
    if (index_iter + 1) % args.eval_freq == 0:
        model.eval()
        eval_NLL, detailed_scores = eval()
        model.train()

        categorical_util.write_dict_to_tensorboard(writer, detailed_scores, base_name="eval", iteration=index_iter+1)

        # If model performed better on validation than any other iteration so far => save it and eventually replace old model
        if eval_NLL < best_save_dict["metric"]:
            best_save_iter = get_checkpoint_filename(index_iter+1)
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
            export_best_results(checkpoint_path, index_iter + 1)
            export_result_txt()
        evaluation_dict[index_iter + 1] = best_save_dict["metric"]

    # Independent of evaluation, the model is saved every "save_freq" steps. This prevents loss of information if model does not improve for a while
    if (index_iter + 1) % args.save_freq == 0 and not os.path.isfile(get_checkpoint_filename(index_iter+1)):
        save_train_model(index_iter + 1)
        if last_save is not None and os.path.isfile(last_save) and last_save != best_save_iter:
            print("Removing checkpoint %s..." % last_save)
            os.remove(last_save)
        last_save = get_checkpoint_filename(index_iter+1)
## End training loop

# Before testing, load best model and check whether its validation performance is in the right range (to prevent major loading issues)
if best_save_iter is not None:
    categorical_util.load_model(best_save_iter, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    eval_NLL, detailed_scores = eval()
    if eval_NLL != best_save_dict["metric"]:
        if abs(eval_NLL - best_save_dict["metric"]) > 1e-1:
            print("[!] WARNING: new evaluation significantly differs from saved one (%s vs %s)! Probably a mistake in the saving/loading part..." % (str(eval_NLL), str(best_save_dict["metric"])))
        else:
            print("[!] WARNING: new evaluation sligthly differs from saved one (%s vs %s)." % (str(eval_NLL), str(best_save_dict["metric"])))
else:
    print("Using last model as no models were saved...")

def test(**kwargs):
    return eval(data_loader=test_data_loader, **kwargs)

def finalize_summary(writer, iteration, checkpoint_path):
    # This function is called after the finishing training and performing testing.
    # Can be used to add something to the summary before finishing.
    pass

# Testing the trained model
test_NLL, detailed_scores = test()
print("="*50+"\nTest performance: %lf" % (test_NLL))
detailed_scores["original_NLL"] = test_NLL
best_save_dict["test"] = detailed_scores
finalize_summary(writer, args.max_iterations, checkpoint_path)

export_result_txt()
writer.close()

# Function for cleaning up the checkpoint directory
def clean_up_dir():
    print("Cleaning up directory " + str(checkpoint_path) + "...")
    for file_in_dir in sorted(glob(os.path.join(checkpoint_path, "*"))):
        print("Removing file " + file_in_dir)
        try:
            if os.path.isfile(file_in_dir):
                os.remove(file_in_dir)
            elif os.path.isdir(file_in_dir): 
                shutil.rmtree(file_in_dir)
        except Exception as e:
            print(e)

if args.clean_up:
    clean_up_dir()
    os.rmdir(checkpoint_path)