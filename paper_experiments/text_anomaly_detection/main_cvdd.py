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
from cvdd_models import *

## 参数设置
parser = argparse.ArgumentParser(description='pytorch cvdd for anomaly detection')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables cuda training')
parser.add_argument(
    '--cuda-device',
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
    '--min_count', 
    type=int, 
    default=3,
    help='min count of words in the dataset')
parser.add_argument(
    '--max_seq', 
    type=int, 
    default=450,
    help='max text length in the dataset')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    help='input batch size for training')
parser.add_argument(
    '--n_attention_heads', 
    type=int, 
    default=3, 
    help='number of attention heads in self-attention module')
parser.add_argument(
    '--attention_size', 
    type=int, 
    default=150, 
    help='self-attention module dimensionality')
parser.add_argument(
    '--pretrain_model',
    default='GloVe_6B',
    help='GloVe_6B | FastText_en')
parser.add_argument(
    '--glove_dim', 
    type=int, 
    default=50, 
    help='dimensionality of Glove 6B')
parser.add_argument(
    '--lr', type=float, default=0.01, help='learning rate')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train')
parser.add_argument(
    '--lambda_p', 
    type=float, 
    default=1.0,
    help='hyperparameter for context vector orthogonality regularization P = (CCT - I)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.cuda_device if args.cuda else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## 数据下载
dataset = getattr(datasets, args.dataset)(normal_class=args.normal_class, min_count=args.min_count, max_seq=args.max_seq)
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
if args.pretrain_model in ['GloVe_6B', 'FastText_en']:
    if args.pretrain_model in ['GloVe_6B']:
        word_vectors = GloVe(name='6B', dim=args.glove_dim, cache='data/word_vectors_cache')
        embedding = MyEmbedding(dataset.encoder.vocab_size, args.glove_dim, update_embedding=False, reduction='none')
    if args.pretrain_model in ['FastText_en']:
        word_vectors = FastText(language='en', cache='data/word_vectors_cache')
        embedding = MyEmbedding(dataset.encoder.vocab_size, 300, update_embedding=False, reduction='none')
    
    # Init embedding with pre-trained word vectors
    for i, token in enumerate(dataset.encoder.vocab):
        embedding.weight.data[i] = word_vectors[token]

model = CVDDNet(embedding, attention_size=args.attention_size, n_attention_heads=args.n_attention_heads)

model.to(device)

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of parameters:',pytorch_total_params)

## 训练及测试
def train():
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        text_batch, _, _ = data
        text_batch = text_batch.to(device)
        # text_batch.shape = (sentence_length, batch_size)

        optimizer.zero_grad()
        cosine_dists, context_weights = model(text_batch)
        scores = context_weights * cosine_dists
        # scores.shape = (batch_size, n_attention_heads)

        # get orthogonality penalty: P = (CCT - I)
        I = torch.eye(args.n_attention_heads).to(device)
        CCT = model.c @ model.c.transpose(1, 2)
        P = torch.mean((CCT.squeeze() - I) ** 2)

        # compute loss
        loss_P = args.lambda_p * P
        loss_emp = torch.mean(torch.sum(scores, dim=1))
        loss = loss_emp + loss_P

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(text_batch.size(1))
        pbar.set_description('Train, loss: {:.6f}'.format(
            train_loss / (batch_idx + 1)))
        
    pbar.close()

def validate(model, loader):
    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        text_batch, _, _ = data
        text_batch = text_batch.to(device)
    
        with torch.no_grad():
            cosine_dists, context_weights = model(text_batch)
            scores = context_weights * cosine_dists

            I = torch.eye(args.n_attention_heads).to(device)
            CCT = model.c @ model.c.transpose(1, 2)
            P = torch.mean((CCT.squeeze() - I) ** 2)

            loss_P = args.lambda_p * P
            loss_emp = torch.mean(torch.sum(scores, dim=1))
            loss = loss_emp + loss_P
            val_loss += loss.item() 

        pbar.update(text_batch.size(1))
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

            cosine_dists, context_weights = model(text_batch)
            ad_scores = torch.mean(cosine_dists, dim=1) 

            label_score += list(zip(label_batch.cpu().data.numpy().tolist(), ad_scores.cpu().data.numpy().tolist()))
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    print('Test set AUC: {:.2f}%'.format(100. * test_auc))
    print('Finished testing.')


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

model.c.data = torch.from_numpy(initialize_context_vectors(model, train_loader, device)[np.newaxis, :]).to(device)  

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
        'Best validation at epoch {}: Average loss: {:.4f}'.
        format(best_validation_epoch, best_validation_loss))

test(best_model, test_loader)