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

import datasets
from models import *

## 参数设置
parser = argparse.ArgumentParser(description='pytorch text flow for anomaly detection')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables cuda training')
parser.add_argument(
    '--device',
    default='cuda:0',
    help='cuda:0 | ...')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed')
parser.add_argument(
    '--dataset',
    default='REUTERS_DATA',
    help='REUTERS_DATA | NEWSGROUP_DATA | IMDB_DATA')
parser.add_argument(
    '--normal_class', 
    type=int, 
    default=0,
    help='specify the normal class of the dataset (all other classes are considered anomalous).')
parser.add_argument(
    '--tokenize',
    default='spacy',
    help='spacy | bert')
parser.add_argument(
    '--use_tfidf_weights',
    action='store_true',
    default=False,
    help='use tfidf weights')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    help='input batch size for training')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=64,
    help='input batch size for testing')
parser.add_argument(
    '--model', default='maf', help='maf | maf-split | maf-split-glow | maf-glow')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=5,
    help='number of invertible blocks')
parser.add_argument(
    '--pretrain_model',
    default='FastText_en',
    help='GloVe_6B | FastText_en | bert')
parser.add_argument(
    '--embedding_reduction',
    default='mean',
    help='mean | max')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of epochs to train')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.cuda_device if args.cuda else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

## 数据下载
dataset = getattr(datasets, args.dataset)(tokenize=args.tokenize, normal_class=args.normal_class, use_tfidf_weights=args.use_tfidf_weights)

def collate_fn(batch):
    """ list of tensors to a batch tensors """
    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

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
valid_sampler = BucketBatchSampler(dataset.valid_set, batch_size=args.test_batch_size, drop_last=False,
                                           sort_key=lambda r: len(r['text']))
test_sampler = BucketBatchSampler(dataset.test_set, batch_size=args.test_batch_size, drop_last=True,
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
if args.pretrain_model in ['GloVe_6B', 'FastText_en']:
    if args.pretrain_model in ['GloVe_6B']:
        word_vectors = GloVe(name='6B', dim=300, cache='data/word_vectors_cache')
    if args.pretrain_model in ['FastText_en']:
        word_vectors = FastText(language='en', cache='data/word_vectors_cache')
    embedding = MyEmbedding(dataset.encoder.vocab_size, 300, update_embedding=True, reduction=args.embedding_reduction, use_tfidf_weights=args.use_tfidf_weights, normalize=True)
    # Init embedding with pre-trained word vectors
    for i, token in enumerate(dataset.encoder.vocab):
        embedding.weight.data[i] = word_vectors[token]
if args.pretrain_model in ['bert']:
    embedding = BERT(update_embedding=True, reduction=args.embedding_reduction, use_tfidf_weights=args.use_tfidf_weights, normalize=True)

num_inputs = embedding.embedding_size
num_hidden = 1024
act = 'relu'

num_cond_inputs = None
modules = []
if args.model == 'maf':
    for _ in range(args.num_blocks):
        modules += [
            MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            BatchNormFlow(num_inputs),
            Reverse(num_inputs)
            ]
elif args.model == 'maf-split':
    for _ in range(args.num_blocks):
        modules += [
            MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            BatchNormFlow(num_inputs),
            Reverse(num_inputs)]
elif args.model == 'maf-glow':
    for _ in range(args.num_blocks):
        modules += [
            MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            BatchNormFlow(num_inputs),
            InvertibleMM(num_inputs)]
elif args.model == 'maf-split-glow':
    for _ in range(args.num_blocks):
        modules += [
            MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            BatchNormFlow(num_inputs),
            InvertibleMM(num_inputs)]

flows = FlowSequential(*modules)
model = ReduceTextFlowModel(embedding, flows)
# print(model)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)

## 训练及测试
def train():
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        text_batch, _, weights = data
        text_batch, weights = text_batch.to(device), weights.to(device)

        optimizer.zero_grad()
        loss = -model(text_batch, weights).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(text_batch.size(1))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))
        
    pbar.close()
        
    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.momentum = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            text_batch, _, weights = data
            text_batch, weights = text_batch.to(device), weights.to(device)
            model(text_batch, weights)

    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.momentum = 1

def validate(model, loader):
    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        text_batch, _, weights = data
        text_batch, weights = text_batch.to(device), weights.to(device)
    
        with torch.no_grad():
            val_loss += -model(text_batch, weights).sum().item() 
        pbar.update(text_batch.size(1))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    pbar.close()
    return val_loss / len(loader.dataset)

def test(model, loader):
    print('Starting testing...')
    model.eval()
    label_score = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            text_batch, label_batch, weights = data
            text_batch, label_batch, weights = text_batch.to(device), label_batch.to(device), weights.to(device)
                
            scores = -model(text_batch, weights)
            ad_scores = scores.squeeze()

            label_score += list(zip(label_batch.cpu().data.numpy().tolist(), ad_scores.cpu().data.numpy().tolist()))
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    scores = np.nan_to_num(scores)
    test_auc = roc_auc_score(labels, scores)
    print('Test set AUC: {:.2f}%'.format(100. * test_auc))
    print('Finished testing.')


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))

    train()
    validation_loss = validate(model, valid_loader)

    if epoch - best_validation_epoch >= 30:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print(
        'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
        format(best_validation_epoch, -best_validation_loss))

test(best_model, test_loader)