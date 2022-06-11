import argparse
import random
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
import torch.optim as optim
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchnlp.word_to_vector import GloVe
from torchtext.vocab import FastText

from cnf.general.train import get_default_train_arguments
from cnf.general.mutils import get_param_val, general_args_to_params, get_device, create_channel_mask, create_optimizer_from_args
from cnf.general.parameter_scheduler import add_scheduler_parameters, scheduler_args_to_params, create_scheduler
from cnf.general.task import TaskTemplate
from cnf.layers.categorical_encoding.mutils import add_encoding_parameters, encoding_args_to_params
from cnf.layers.flows.distributions import add_prior_distribution_parameters, prior_distribution_args_to_params, create_prior_distribution
from cnf.experiments.language_modeling.task import TaskLanguageModeling
from cnf.experiments.language_modeling.flow_model import FlowLanguageModeling

import datasets

## 参数设置
parser = get_default_train_arguments()
# Add parameters for prior distribution
add_prior_distribution_parameters(parser)
# Add parameters for categorical encoding
add_encoding_parameters(parser)
# Parameter for schedulers
add_scheduler_parameters(parser, ["beta"], {"beta": ("exponential", 1.0, 2.0, 5000, 2, 0)})

# Parse given parameters and start training
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.cuda_device if args.cuda else "cpu")

def args_to_params(args):
	model_params, optimizer_params = general_args_to_params(args, model_params=dict())
	model_params["prior_distribution"] = prior_distribution_args_to_params(args)
	model_params["categ_encoding"] = encoding_args_to_params(args)
	sched_params = scheduler_args_to_params(args, ["beta"])
	model_params.update(sched_params)
	dataset_params = {
		"max_seq_len": args.max_seq_len,
		"dataset": args.dataset,
		"use_rnn": False
	}
	coupling_params = {p_name: getattr(args, p_name) for p_name in vars(args) if p_name.startswith("coupling_")}
	model_params.update(coupling_params)
	model_params.update(dataset_params)
	return model_params, optimizer_params

model_params, optimizer_params = args_to_params(args)

## 数据下载
dataset = getattr(datasets, args.dataset)(normal_class=args.normal_class, min_count=args.min_count, max_seq=args.max_seq_len)
print("vocab_size={}".format(dataset.encoder.vocab_size))

def collate_fn(batch):
    """ list of tensors to a batch tensors """
    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t().contiguous())

    text_batch, _ = stack_and_pad_tensors([row['text'] for row in batch])
    label_batch = torch.stack([row['label'] for row in batch])
    weights = [row['weight'] for row in batch]
    # check if weights are empty
    if weights[0].nelement() == 0:
        weight_batch = torch.empty(0)
    else:
        weight_batch, _ = stack_and_pad_tensors([row['weight'] for row in batch])
        weight_batch = transpose(weight_batch)

    return transpose(text_batch), label_batch.float(), weight_batch

train_sampler = BucketBatchSampler(dataset.train_set, batch_size=args.batch_size, drop_last=True,
                                           sort_key=lambda r: len(r['text']))
valid_sampler = BucketBatchSampler(dataset.valid_set, batch_size=args.batch_size, drop_last=False,
                                           sort_key=lambda r: len(r['text']))
test_sampler = BucketBatchSampler(dataset.test_set, batch_size=args.batch_size, drop_last=True,
                                          sort_key=lambda r: len(r['text']))

train_loader = torch.utils.data.DataLoader(
    dataset=dataset.train_set, batch_sampler=train_sampler, collate_fn=collate_fn)

valid_loader = torch.utils.data.DataLoader(
    dataset=dataset.valid_set,
    batch_sampler=valid_sampler,
    collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(
    dataset=dataset.test_set,
    batch_sampler=test_sampler,
    collate_fn=collate_fn)

## 模型及优化器
model = FlowLanguageModeling(model_params=model_params, vocab_size=dataset.encoder.vocab_size, vocab=None, dataset_class=None)
model = model.to(device)

prior_dist_params = get_param_val(model_params, "prior_distribution", allow_default=False, error_location="TaskLanguageModeling - init")
prior_distribution = create_prior_distribution(prior_dist_params)

beta_scheduler = create_scheduler(model_params["beta"], "beta")

parameters_to_optimize = model.parameters()
optimizer = create_optimizer_from_args(parameters_to_optimize, optimizer_params)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, optimizer_params["lr_decay_step"], gamma=optimizer_params["lr_decay_factor"])
lr_minimum = optimizer_params["lr_minimum"]

## 训练及测试
def _calc_loss(neg_ldj, neglog_prob, x_length):
    neg_ldj = (neg_ldj / x_length.float())
    neglog_prob = (neglog_prob / x_length.float())
    loss = neg_ldj + neglog_prob

    loss_mean = loss.mean()
    neg_ldj = neg_ldj.mean()
    neglog_prob = neglog_prob.mean()
    return loss_mean, neg_ldj, neglog_prob, loss

def _preprocess_batch(batch):
    x_in = batch
    x_length = x_in.new_zeros(x_in.size(0), dtype=torch.long) + x_in.size(1)
    x_channel_mask = x_in.new_ones(x_in.size(0), x_in.size(1), 1, dtype=torch.float32)
    return x_in, x_length, x_channel_mask

def _train_batch_flow(x_in, x_length, x_channel_mask, iteration=0, **kwargs):
    z, ldj, ldj_per_layer = model(x_in, reverse=False, get_ldj_per_layer=True, 
                                        beta=beta_scheduler.get(iteration),
                                        length=x_length)
    neglog_prob = -(prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
    neg_ldj = -ldj
    
    loss, neg_ldj, neglog_prob, _ = _calc_loss(neg_ldj, neglog_prob, x_length)

    return loss

def _train_batch(batch, iteration):
    x_in, x_length, x_channel_mask = _preprocess_batch(batch)
    return _train_batch_flow(x_in, x_length, x_channel_mask, iteration)

def train(iteration):
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        text_batch, _, _ = data
        text_batch = text_batch.to(device)
        # text_batch.shape = (sentence_length, batch_size)
        text_batch = text_batch.transpose(0, 1)

        loss = _train_batch(text_batch, iteration)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters_to_optimize, 0.25)
        optimizer.step()
        if optimizer.param_groups[0]['lr'] > lr_minimum:
            lr_scheduler.step()

        train_loss += loss.item()

        pbar.update(text_batch.size(0))
        pbar.set_description('Train, loss: {:.6f}'.format(
            train_loss / (batch_idx + 1)))

        iteration += 1
        
    pbar.close()
    return iteration

def _eval_batch(batch, is_test=False):
    x_in, x_length, x_channel_mask = _preprocess_batch(batch)
    return _eval_batch_flow(x_in, x_length, x_channel_mask, is_test=is_test)

def _eval_batch_flow(x_in, x_length, x_channel_mask, is_test=False, **kwargs):
    z, ldj, ldj_per_layer = model(x_in, reverse=False, get_ldj_per_layer=True, 
                                        length=x_length)
    neglog_prob = -(prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
    neg_ldj = -ldj
    loss, _, _ , loss_batch= _calc_loss(neg_ldj, neglog_prob, x_length)
    return loss, loss_batch

def validate(model, loader):
    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        text_batch, _, _ = data
        text_batch = text_batch.to(device)
        text_batch = text_batch.transpose(0, 1)
    
        with torch.no_grad():
            loss, _ = _eval_batch(text_batch, is_test=False)
            val_loss += loss.item() 

        pbar.update(text_batch.size(0))
        pbar.set_description('Val, loss: {:.6f}'.format(
            val_loss / (batch_idx + 1)))

    pbar.close()
    return val_loss /  (batch_idx + 1)

def test(model, loader):
    print('Starting testing...')
    model.eval()
    label_score = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            text_batch, label_batch, _ = data
            text_batch, label_batch = text_batch.to(device), label_batch.to(device)
            text_batch = text_batch.transpose(0, 1)

            _, loss = _eval_batch(text_batch, is_test=True)

            label_score += list(zip(label_batch.cpu().data.numpy().tolist(), loss.cpu().data.numpy().tolist()))
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    print('Test set AUC: {:.2f}%'.format(100. * test_auc))
    print('Finished testing.')


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

iteration = 0
for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))

    iteration = train(iteration)
    validation_loss = validate(model, valid_loader)

    if epoch - best_validation_epoch >= 30:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print(
        'Best validation at epoch {}: Average loss: {:.4f}'.
        format(best_validation_epoch, best_validation_loss))

test(best_model, test_loader)