import argparse
import random
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text.text_encoder import pad_tensor

from discrete_flows.disc_models import *

import datasets

## 参数设置
parser = argparse.ArgumentParser(description='pytorch discrete flows for anomaly detection')
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
    '--lr', type=float, default=0.01, help='learning rate')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train')
parser.add_argument(
    '--disc_layer_type',
    default='autoreg',
    help='autoreg | bipartite')

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

    padded = [pad_tensor(tensor, args.max_seq) for tensor in [row['text'] for row in batch]]
    text_batch = torch.stack(padded, dim=0).contiguous()
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
vocab_size = dataset.encoder.vocab_size
num_flows = 1
temperature = 0.1
vector_length = vocab_size*args.max_seq

flows = []
for i in range(num_flows):
    if args.disc_layer_type == 'autoreg':
        
        layer = torch.nn.Embedding(vocab_size, vocab_size)
        disc_layer = DiscreteAutoregressiveFlow(layer, temperature, vocab_size)
    
    elif args.disc_layer_type == 'bipartite':
        
        layer = torch.nn.Embedding(vector_length//2, vector_length//2)
        disc_layer = DiscreteBipartiteFlow(layer, i%2, temperature, vocab_size, vector_length, embedding=True)
    
    flows.append(disc_layer)
    
model = DiscreteAutoFlowModel(flows)
model.to(device)

base_log_probs = torch.tensor(torch.randn(args.max_seq, vocab_size)).to(device)
base_log_probs.requires_grad = True

optimizer = torch.optim.Adam( 
        [
            {'params': model.parameters() , 'lr':args.lr},
            {'params': base_log_probs, 'lr': args.lr }
        ])

## 训练及测试
def train(iteration):
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        text_batch, _, _ = data
        text_batch = text_batch.to(device)
        # text_batch.shape = (sentence_length, batch_size)
        x = text_batch.transpose(0, 1)
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        if args.disc_layer_type == 'bipartite':
            x = F.one_hot(x, num_classes = vocab_size).float()
            x= x.view(x.shape[0], -1) #flattening vector

        optimizer.zero_grad()
        zs = model.forward(x)
        
        if args.disc_layer_type == 'bipartite':
            zs = zs.view(batch_size, sequence_length, -1) # adding back in sequence dimension
        
        base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1)
        logprob = zs*base_log_probs_sm # zs are onehot so zero out all other logprobs. 
        loss = -torch.sum(logprob)/batch_size/sequence_length
        
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(text_batch.size(1))
        pbar.set_description('Train, loss: {:.6f}'.format(
            train_loss / (batch_idx + 1)))

        iteration += 1
        
    pbar.close()
    return iteration

def validate(model, base_log_probs, loader):
    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        text_batch, _, _ = data
        text_batch = text_batch.to(device)
        x = text_batch.transpose(0, 1)
    
        with torch.no_grad():
            batch_size = x.shape[0]
            sequence_length = x.shape[1]

            if args.disc_layer_type == 'bipartite':
                x = F.one_hot(x, num_classes = vocab_size).float()
                x= x.view(x.shape[0], -1) #flattening vector

            zs = model.forward(x)
            
            if args.disc_layer_type == 'bipartite':
                zs = zs.view(batch_size, sequence_length, -1) # adding back in sequence dimension
            
            base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1)
            logprob = zs*base_log_probs_sm # zs are onehot so zero out all other logprobs. 
            loss = -torch.sum(logprob)/batch_size/sequence_length
            val_loss += loss.item() 

        pbar.update(text_batch.size(1))
        pbar.set_description('Val, loss: {:.6f}'.format(
            val_loss / (batch_idx + 1)))

    pbar.close()
    return val_loss /  (batch_idx + 1)

def test(model, base_log_probs, loader):
    print('Starting testing...')
    model.eval()
    label_score = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            text_batch, label_batch, _ = data
            text_batch, label_batch = text_batch.to(device), label_batch.to(device)
            x = text_batch.transpose(0, 1)

            batch_size = x.shape[0]
            sequence_length = x.shape[1]

            if args.disc_layer_type == 'bipartite':
                x = F.one_hot(x, num_classes = vocab_size).float()
                x= x.view(x.shape[0], -1) #flattening vector

            zs = model.forward(x)
            
            if args.disc_layer_type == 'bipartite':
                zs = zs.view(batch_size, sequence_length, -1) # adding back in sequence dimension
            
            base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1)
            logprob = zs*base_log_probs_sm # zs are onehot so zero out all other logprobs. 
            logprob = logprob.sum(dim=[1,2])
            loss = -logprob/sequence_length

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
best_base = base_log_probs

iteration = 0
for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))

    iteration = train(iteration)
    validation_loss = validate(model, base_log_probs, valid_loader)

    if epoch - best_validation_epoch >= 30:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)
        best_base = copy.deepcopy(base_log_probs)

    print(
        'Best validation at epoch {}: Average loss: {:.4f}'.
        format(best_validation_epoch, best_validation_loss))

test(best_model, best_base, test_loader)