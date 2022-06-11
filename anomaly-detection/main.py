import argparse
from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim

import datasets
from model import *

## Training settings
parser = argparse.ArgumentParser(description='PyTorch anomaly detection using flows')
parser.add_argument(
    '--dataset',
    default='Mnist',
    help='Mnist | Cifar10')
parser.add_argument(
    '--model', default='realnvp', help='realnvp | maf | glow | svdd')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed')
parser.add_argument(
    '--batch-size',
    type=int,
    default=200,
    help='input batch size for training')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=200,
    help='input batch size for testing')
parser.add_argument(
    '--epochs',
    type=int,
    default=150,
    help='number of epochs to train')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--normal-class', type=int, default=0, help='specify the normal class of the dataset(all other classes are considered anomalous)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

## Load dataset
dataset = getattr(datasets, args.dataset)(args.normal_class)

train_loader = torch.utils.data.DataLoader(
    dataset.train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    dataset.test_set,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

## Model settings
if args.dataset == "Mnist":
    num_inputs = 784 
elif args.dataset == "Cifar10":
    num_inputs = 3072 
num_cond_inputs = None
modules = []
if args.model == 'realnvp':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(5):
        modules += [
            CouplingLayer(
                num_inputs, 1024, mask, num_cond_inputs,
                s_act='tanh', t_act='relu'),
            BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask
if args.model == 'glow':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(5):
        modules += [
            BatchNormFlow(num_inputs),
            LUInvertibleMM(num_inputs),
            CouplingLayer(
                num_inputs, 1024, mask, num_cond_inputs,
                s_act='tanh', t_act='relu')
        ]
        mask = 1 - mask
if args.model == 'maf':
    for _ in range(5):
        modules += [
            MADE(num_inputs, 1024, num_cond_inputs, act='relu'),
            BatchNormFlow(num_inputs),
            Reverse(num_inputs)
        ]

model = FlowSequential(*modules)
for module in model.modules(): 
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)

## Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=(50,), gamma=0.1)

## Training and testing
def train(epoch):
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        inputs, _ = data
        inputs = inputs.reshape(len(inputs), -1)
        data = inputs.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train loss: {:.6f}'.format(
            train_loss / (batch_idx + 1)))

    scheduler.step()

    pbar.close()


def test():
    print('Starting testing...')
    model.eval()
    label_score = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.reshape(len(inputs), -1)
            data = inputs.to(device)
            scores = -model.log_probs(data)
            scores = scores.squeeze()
            scores = scores.cpu().data.numpy().tolist()

            label_score += list(zip(labels.cpu().data.numpy().tolist(), scores))
    
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    print('Test set AUC: {:.2f}%'.format(100. * test_auc))
    print('Finished testing.')

for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))
    train(epoch)

test()