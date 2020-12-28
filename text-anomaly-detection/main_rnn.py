import argparse
import numpy as np
import time
import math
import os
import hashlib

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import datasets
from temp_models import *
import util

## 参数设置
parser = argparse.ArgumentParser(description='pytorch text anomaly detection using rnn language model')
parser.add_argument(
    '--dataset',
    type=str, 
    default='penn',
    help='penn | pennchar')
parser.add_argument(
    '--wikitext_char', action='store_true', default=False, help='Load character-level WikiText. Use when in-dist is character-level. Dictionary also uses in-dist')
parser.add_argument(
    '--model', 
    type=str, 
    default='LSTM', 
    help='LSTM | QRNN | GRU')
parser.add_argument(
    '--use_OE', action='store_true', default=False, help='outlier exposure')
parser.add_argument(
    '--cuda-device',
    type=str, 
    default='cuda:0',
    help='cuda:0 | ...')
parser.add_argument(
    '--when', nargs="+", type=int, default=[-1], help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument(
    '--resume_oe', type=str,  default='', help='path of model to resume for OE: output/.../model.pt')
parser.add_argument(
    '--resume_ood', type=str,  default='', help='path of model to resume for OOD: output/.../model.pt')

parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disable cuda training')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed')
parser.add_argument(
    '--nonmono', type=int, default=5, help='random seed for validation loss')
parser.add_argument(
    '--log-interval', type=int, default=200, help='report interval')

parser.add_argument(
    '--batch_size',
    type=int,
    default=80,
    help='input batch size for training')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=10,
    help='input batch size for validating')
parser.add_argument(
    '--test_batch_size',
    type=int,
    default=1,
    help='input batch size for testing')
parser.add_argument(
    '--bptt', type=int, default=70, help='sequence length')

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
    '--tied',
    action='store_false',
    default=True,
    help='enable tied weights')

parser.add_argument(
    '--lr', type=float, default=30, help='initial learning rate')
parser.add_argument(
    '--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument(
    '--epochs', type=int, default=8000, help='upper epoch limit')
parser.add_argument(
    '--alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument(
    '--beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument(
    '--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
parser.add_argument(
    '--optimizer', type=str,  default='sgd', help='optimizer to use (sgd, adam)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.cuda_device if args.cuda else "cpu")

setattr(args, 'save', 'output/'+args.model+'-'+args.dataset+'-'+str(args.use_OE)+'OE')

os.makedirs(args.save, exist_ok=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

## 数据下载
def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)


fn = 'datacache/corpus.{}.data'.format(hashlib.md5(args.dataset.encode()).hexdigest())
# if os.path.exists(fn):
#     print('Loading cached in-dist dataset...')
#     dataset = torch.load(fn)
# else:
#     print('Producing in-dist dataset...')
#     dataset = getattr(datasets, "LM_DATA")(args.dataset)
#     torch.save(dataset, fn)
print('Producing in-dist dataset...')
dataset = getattr(datasets, "LM_DATA")(args.dataset)
torch.save(dataset, fn)

print(dataset.vocab_size, dataset.trn.x.shape, dataset.val.x.shape, dataset.tst.x.shape)

train_data = util.batchify(dataset.trn.x, args.batch_size, device)
val_data = util.batchify(dataset.val.x, args.eval_batch_size, device)
test_data = util.batchify(dataset.tst.x, args.test_batch_size, device)

print(train_data.shape, val_data.shape, test_data.shape)

# Load OE data
print('Producing OE dataset...')
if args.wikitext_char:
    oe_dataset = util.CorpusWikiTextChar('data/wikitext-2', dataset.dictionary)

    oe_train_dataset = util.batchify(oe_dataset.train, args.batch_size, device)
    oe_val_dataset = util.batchify(oe_dataset.valid, args.eval_batch_size, device)
else:
    oe_dataset = getattr(datasets, "LM_DATA")('wikitext-2', dataset.dictionary)

    oe_train_dataset = util.batchify(oe_dataset.trn.x, args.batch_size, device)
    oe_val_dataset = util.batchify(oe_dataset.val.x, args.eval_batch_size, device)

## 模型及优化器
ntokens = len(dataset.dictionary)
model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

if args.resume_oe:
    print('Resuming model for OE...')
    model_load(args.resume_oe)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop

splits = []
if ntokens > 500000:
    # One Billion
    # This produces fairly even matrix mults for the buckets:
    # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
    splits = [4200, 35000, 180000]
elif ntokens > 75000:
    # WikiText-103
    splits = [2800, 20000, 76000]
print('Using splits', splits)
criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)

model = model.to(device)
criterion = criterion.to(device)

params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

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
    total_oe_loss = 0
    hidden = model.init_hidden(2*batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = util.get_batch(data_source, i, args, evaluation=True)
        data_oe, _ = util.get_batch(oe_val_dataset, i, args, evaluation=True)

        assert len(data.size()) != 1
        assert len(data_oe.size()) != 1

        if data.size(0) != data_oe.size(0):
            continue

        output, hidden, rnn_hs, dropped_rnn_hs = model(torch.cat([data, data_oe], dim=1), hidden, return_h=True)
        output, output_oe = torch.chunk(dropped_rnn_hs[-1], dim=1, chunks=2)
        output, output_oe = output.contiguous(), output_oe.contiguous()
        output = output.view(output.size(0)*output.size(1), output.size(2))

        loss = criterion(model.decoder.weight, model.decoder.bias, output, targets).data

        # OE loss
        logits_oe = model.decoder(output_oe)
        smaxes_oe = F.softmax(logits_oe - torch.max(logits_oe, dim=-1, keepdim=True)[0], dim=-1)
        loss_oe = -smaxes_oe.log().mean(-1)
        loss_oe = loss_oe.mean().data

        total_loss += len(data) * loss
        total_oe_loss += len(data) * loss_oe

        hidden = util.repackage_hidden(hidden)
    return total_loss.item() / len(data_source), total_oe_loss.item() / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    total_oe_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(2*args.batch_size)
    batch = 0 
    seq_len = args.bptt

    for i in range(0, train_data.size(0), args.bptt): 

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = util.get_batch(train_data, i, args, seq_len=seq_len)
        data_oe, _ = util.get_batch(oe_train_dataset, i, args, seq_len=seq_len)

        if data.size(0) != data_oe.size(0):  # Don't train on this batch if the sequence lengths are different (happens at end of epoch).
            continue

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = util.repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(torch.cat([data, data_oe], dim=1), hidden, return_h=True)
        output, output_oe = torch.chunk(dropped_rnn_hs[-1], dim=1, chunks=2)
        output, output_oe = output.contiguous(), output_oe.contiguous()
        output = output.view(output.size(0)*output.size(1), output.size(2))

        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

        # OE loss
        logits_oe = model.decoder(output_oe)
        smaxes_oe = F.softmax(logits_oe - torch.max(logits_oe, dim=-1, keepdim=True)[0], dim=-1)
        loss_oe = -smaxes_oe.log().mean(-1)  
        loss_oe = loss_oe.mean() 

        if args.use_OE:
            loss_bp = loss + 0.5 * loss_oe
        else:
            loss_bp = loss

        loss_bp.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        total_oe_loss += loss_oe.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            cur_oe_loss = total_oe_loss.item() /args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | oe_loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, cur_oe_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            total_oe_loss = 0
            start_time = time.time()
        
        batch += 1

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
            if prm in optimizer.state.keys():
                # tmp[prm] = prm.data.clone()
                tmp[prm] = prm.data.detach()
                # prm.data = optimizer.state[prm]['ax'].clone()
                prm.data = optimizer.state[prm]['ax'].detach()

        val_loss2, val_oe_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | val oe_loss {:5.2f} | '
            'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, val_oe_loss, math.exp(val_loss2), val_loss2 / math.log(2)))
        print('-' * 89)

        if val_loss2 < stored_loss:
            model_save(args.save+'/model.pt')
            print('Saving Averaged!')
            stored_loss = val_loss2

        for prm in model.parameters():
            if prm in tmp.keys():
                # prm.data = tmp[prm].clone()
                prm.data = tmp[prm].detach()
                prm.requires_grad = True

        del tmp

    else:
        val_loss, val_oe_loss = evaluate(val_data, args.eval_batch_size)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | val oe_loss {:5.2f} | '
            'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
            epoch, (time.time() - epoch_start_time), val_loss, val_oe_loss, math.exp(val_loss), val_loss / math.log(2)))
        print('-' * 89)

        if val_loss < stored_loss:
            model_save(args.save+'/model.pt')
            print('Saving model (new best validation)')
            stored_loss = val_loss

        if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
            print('Switching to ASGD')
            optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

        if epoch in args.when:
            print('Saving model before learning rate decreased')
            model_save('{}/model.e{}'.format(args.save, epoch))
            print('Dividing learning rate by 10')
            optimizer.param_groups[0]['lr'] /= 10.

        best_val_loss.append(val_loss)

# Load the best saved model.
model_load(args.save+'/model.pt')


print('Producing ood datasets...')

answers_corpus = util.OODCorpus('data/eng_web_tbk/answers-dev.conllu', dataset.dictionary, char_level=args.wikitext_char)
answers_data = util.batchify(answers_corpus.data, args.test_batch_size, device)

email_corpus = util.OODCorpus('data/eng_web_tbk/email-dev.conllu', dataset.dictionary, char_level=args.wikitext_char)
email_data = util.batchify(email_corpus.data, args.test_batch_size, device)

newsgroup_corpus = util.OODCorpus('data/eng_web_tbk/newsgroup-dev.conllu', dataset.dictionary, char_level=args.wikitext_char)
newsgroup_data = util.batchify(newsgroup_corpus.data, args.test_batch_size, device)

reviews_corpus = util.OODCorpus('data/eng_web_tbk/reviews-dev.conllu', dataset.dictionary, char_level=args.wikitext_char)
reviews_data = util.batchify(reviews_corpus.data, args.test_batch_size, device)

weblog_corpus = util.OODCorpus('data/eng_web_tbk/weblog-dev.conllu', dataset.dictionary, char_level=args.wikitext_char)
weblog_data = util.batchify(weblog_corpus.data, args.test_batch_size, device)

assert args.resume_ood, 'must provide a --resume_ood argument'
print('Resuming model for OOD ...')
model_load(args.resume_ood)
optimizer.param_groups[0]['lr'] = args.lr
model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
if args.wdrop:
    for rnn in model.rnns:
        if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
        elif rnn.zoneout > 0: rnn.zoneout = args.wdrop

model = model.to(device)
criterion = criterion.to(device)

ood_num_examples = test_data.size(0) // 5
recall_level = 0.9

def evaluate_ood(data_source, dataset, batch_size=10, ood=False):
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    losses = []
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        if (i >= ood_num_examples // args.test_batch_size) and (ood is True):
            break

        data, targets = util.get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)

        logits = model.decoder(output)
        smaxes = F.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
        tmp = smaxes[range(targets.size(0)), targets]
        log_prob = torch.log(tmp).mean(0)  # divided by seq len, so this is the negative nats per char
        loss = -log_prob.data.cpu().numpy()
        
        total_loss += loss
        losses.append(loss)

        hidden = util.repackage_hidden(hidden)
    return total_loss.item() / (len(data_source) // args.bptt), losses


# Run on test data.
print('\nPTB')
test_loss, test_losses = evaluate_ood(test_data, dataset, args.test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)


print('\nAnswers (OOD)')
ood_loss, ood_losses = evaluate_ood(answers_data, answers_corpus, args.test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
util.show_performance(ood_losses, test_losses, recall_level=recall_level)


print('\nEmail (OOD)')
ood_loss, ood_losses = evaluate_ood(email_data, email_corpus, args.test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
util.show_performance(ood_losses, test_losses, recall_level=recall_level)


print('\nNewsgroup (OOD)')
ood_loss, ood_losses = evaluate_ood(newsgroup_data, newsgroup_corpus, args.test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
util.show_performance(ood_losses, test_losses, recall_level=recall_level)


print('\nReviews (OOD)')
ood_loss, ood_losses = evaluate_ood(reviews_data, reviews_corpus, args.test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
util.show_performance(ood_losses, test_losses, recall_level=recall_level)


print('\nWeblog (OOD)')
ood_loss, ood_losses = evaluate_ood(weblog_data, weblog_corpus, args.test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
util.show_performance(ood_losses, test_losses, recall_level=recall_level)


