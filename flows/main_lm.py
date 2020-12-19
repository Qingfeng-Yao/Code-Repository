import argparse
import numpy as np
import time
import math
import os
import hashlib

import torch

import datasets
from models import *
import util

## 参数设置
parser = argparse.ArgumentParser(description='pytorch language model')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables cuda training')
parser.add_argument(
    '--cuda-device',
    type=str, 
    default='cuda:0',
    help='cuda:0 | ...')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed')
parser.add_argument(
    '--nonmono', type=int, default=5, help='random seed')
parser.add_argument(
    '--log-interval', type=int, default=200, help='report interval')
parser.add_argument(
    '--dataset',
    type=str, 
    default='PTB_DATA',
    help='PTB_DATA')
parser.add_argument(
    '--model', 
    type=str, 
    default='LSTM', 
    help='LSTM | QRNN | GRU')
parser.add_argument(
    '--emsize', 
    type=int, 
    default=400,
    help='size of word embeddings')
parser.add_argument(
    '--nhid', type=int, default=1150, help='number of hidden units per layer')
parser.add_argument(
    '--nlayers', type=int, default=3, help='number of layers')
parser.add_argument(
    '--bptt', type=int, default=70, help='sequence length')
parser.add_argument(
    '--batch-size',
    type=int,
    default=80,
    help='input batch size for training')
parser.add_argument(
    '--lr', type=float, default=30, help='initial learning rate')
parser.add_argument(
    '--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument(
    '--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
parser.add_argument(
    '--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
parser.add_argument(
    '--dropouti', type=float, default=0.65, help='dropout for input embedding layers (0 = no dropout')
parser.add_argument(
    '--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument(
    '--wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument(
    '--alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument(
    '--beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument(
    '--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
parser.add_argument(
    '--optimizer', type=str,  default='sgd', help='optimizer to use (sgd, adam)')
parser.add_argument(
    '--epochs', type=int, default=8000, help='upper epoch limit')
parser.add_argument(
    '--save', type=str,  default='PTB.pt', help='path to save the final model')
parser.add_argument(
    '--when', nargs="+", type=int, default=[-1], help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

args = parser.parse_args()
args.tied = True
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.cuda_device if args.cuda else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

## 数据下载
fn = 'corpus.{}.data'.format(hashlib.md5(args.dataset.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    dataset = torch.load(fn)
else:
    print('Producing dataset...')
    dataset = getattr(datasets, args.dataset)()
    torch.save(dataset, fn)

print(dataset.vocab_size, dataset.trn.x.shape, dataset.val.x.shape, dataset.tst.x.shape)

eval_batch_size = 10
test_batch_size = 1
train_data = util.batchify(dataset.trn.x, args.batch_size, device)
val_data = util.batchify(dataset.val.x, eval_batch_size, device)
test_data = util.batchify(dataset.tst.x, test_batch_size, device)

print(train_data.shape, val_data.shape, test_data.shape)

## 模型及优化器
ntokens = len(dataset.dictionary)
model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

splits = []
if ntokens > 500000:
    # One Billion
    # This produces fairly even matrix mults for the buckets:
    # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
    splits = [4200, 35000, 180000]
elif ntokens > 75000:
    # WikiText-103
    splits = [2800, 20000, 76000]
print('Using', splits)
criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)

model = model.to(device)
criterion = criterion.(device)

params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

optimizer = None
# Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

## 训练及测试
def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(dataset.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = util.get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = util.repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(dataset.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = util.get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = util.repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    if 't0' in optimizer.param_groups[0]:
        tmp = {}
        for prm in model.parameters():
            tmp[prm] = prm.data.clone()
            prm.data = optimizer.state[prm]['ax'].clone()

        val_loss2 = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
        print('-' * 89)

        if val_loss2 < stored_loss:
            model_save(args.save)
            print('Saving Averaged!')
            stored_loss = val_loss2

        for prm in model.parameters():
            prm.data = tmp[prm].clone()

    else:
        val_loss = evaluate(val_data, eval_batch_size)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
            epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
        print('-' * 89)

        if val_loss < stored_loss:
            model_save(args.save)
            print('Saving model (new best validation)')
            stored_loss = val_loss

        if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
            print('Switching to ASGD')
            optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

        if epoch in args.when:
            print('Saving model before learning rate decreased')
            model_save('{}.e{}'.format(args.save, epoch))
            print('Dividing learning rate by 10')
            optimizer.param_groups[0]['lr'] /= 10.

        best_val_loss.append(val_loss)

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)