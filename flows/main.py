import argparse
import numpy as np
from tqdm import tqdm
import copy

import torch
import torch.optim as optim

import datasets
from models import *

## 参数设置
parser = argparse.ArgumentParser(description='pytorch flows')
parser.add_argument(
    '--dataset',
    type=str, 
    default='MNIST_DATA',
    help='MNIST_DATA')
parser.add_argument(
    '--model', type=str, default='maf', help='flow to use: maf')
parser.add_argument(
    '--cuda-device',
    type=str,
    default='cuda:0',
    help='cuda:0 | ...')

parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables cuda training')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed')

parser.add_argument(
    '--batch-size',
    type=int,
    default=100,
    help='input batch size for training')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    help='input batch size for testing')

parser.add_argument(
    '--num-hidden',
    type=int,
    default=1024,
    help='dim of hidden layer: mnist/1024 | cifar/1024')
parser.add_argument(
    '--n-hidden',
    type=int,
    default=1,
    help='number of hidden layers')
parser.add_argument(
    '--act',
    type=str,
    default='relu',
    help='activation functions')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=5,
    help='number of invertible blocks')
parser.add_argument(
    '--cond',
    action='store_true',
    default=False,
    help='train class conditional flow')

parser.add_argument(
    '--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of epochs to train')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.cuda_device if args.cuda else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.cuda:
    kwargs = {'num_workers': 4, 'pin_memory': True} 
else:
    kwargs = {}

## 数据下载
dataset = getattr(datasets, args.dataset)()
print(dataset.trn.x.shape, dataset.val.x.shape, dataset.tst.x.shape)

if args.cond:
    train_tensor = torch.from_numpy(dataset.trn.x)
    train_labels = torch.from_numpy(dataset.trn.y)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_labels = torch.from_numpy(dataset.val.y)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor, valid_labels)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_labels = torch.from_numpy(dataset.tst.y)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
    num_cond_inputs = 10
else:
    train_tensor = torch.from_numpy(dataset.trn.x)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    num_cond_inputs = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

## 模型及优化器
num_inputs = dataset.n_dims
num_hidden = args.num_hidden
n_hidden = args.n_hidden
act = args.act

modules = []
if args.model == 'maf':
    for _ in range(args.num_blocks):
        modules += [
            MADE(num_inputs, num_hidden, n_hidden, num_cond_inputs, act=act),
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

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of parameters:',pytorch_total_params)

## 训练及测试
def train():
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]

        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))

    pbar.close()
        
    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device),
                train_loader.dataset.tensors[1].to(data.device).float())
    else:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device))

    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.momentum = 1

def validate(model, loader):
    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            val_loss += -model.log_probs(data, cond_data).sum().item()
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    pbar.close()
    return val_loss / len(loader.dataset)

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

validate(best_model, test_loader)
