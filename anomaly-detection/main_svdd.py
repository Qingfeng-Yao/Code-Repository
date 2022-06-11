import argparse
from tqdm import tqdm
import numpy as np
  
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim

import datasets
from model import *
  
## training settings
parser = argparse.ArgumentParser(description='PyTorch anomaly detection using svdd')
parser.add_argument(
    '--dataset',
    default='Mnist',
    help='Mnist | Cifar10')
parser.add_argument(
    '--model', default='mnist_lenet', help='mnist_lenet | cifar10_lenet')
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
if args.model == 'mnist_lenet':
    model = MNIST_LeNet()
    ### model pretrain
    ae_net = MNIST_LeNet_Autoencoder()
    ae_net.to(device)
    ae_optimizer = optim.Adam(ae_net.parameters(), lr=0.0001, weight_decay=0.5e-3)
    ae_scheduler = optim.lr_scheduler.MultiStepLR(ae_optimizer, milestones=(50,), gamma=0.1)
    ae_net.train()
    for epoch in range(150):
        if epoch in (50,):
            print(' LR scheduler: new learning rate is %g' % float(ae_scheduler.get_lr()[0]))
        loss_epoch = 0.0
        n_batches = 0
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)

            ae_optimizer.zero_grad()
            outputs = ae_net(inputs)
            scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
            loss = torch.mean(scores)
            loss.backward()
            ae_optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1
        ae_scheduler.step()

    print('Finished pretraining.')
    ### init_network_weights_from_pretraining
    model_dict = model.state_dict()
    ae_net_dict = ae_net.state_dict()
    ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in model_dict} # Filter out decoder network keys
    model_dict.update(ae_net_dict)
    model.load_state_dict(model_dict)
if args.model == 'cifar10_lenet':
    model = CIFAR10_LeNet()
    ### model pretrain
    ae_net = CIFAR10_LeNet_Autoencoder()
    ae_net.to(device)
    ae_optimizer = optim.Adam(ae_net.parameters(), lr=0.0001, weight_decay=0.5e-6)
    ae_scheduler = optim.lr_scheduler.MultiStepLR(ae_optimizer, milestones=(250,), gamma=0.1)
    ae_net.train()
    for epoch in range(350):
        if epoch in (50,):
            print(' LR scheduler: new learning rate is %g' % float(ae_scheduler.get_lr()[0]))
        loss_epoch = 0.0
        n_batches = 0
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)

            ae_optimizer.zero_grad()
            outputs = ae_net(inputs)
            scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
            loss = torch.mean(scores)
            loss.backward()
            ae_optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1
        ae_scheduler.step()

    print('Finished pretraining.')
    ### init_network_weights_from_pretraining
    model_dict = model.state_dict()
    ae_net_dict = ae_net.state_dict()
    ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in model_dict} # Filter out decoder network keys
    model_dict.update(ae_net_dict)
    model.load_state_dict(model_dict)

model.to(device)

## Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.5e-6)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=(50,), gamma=0.1)

## Training and testing
def train(epoch):
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        inputs, _ = data
        data = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        dist = torch.sum((outputs - c) ** 2, dim=1)
        loss = torch.mean(dist)
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
            data = inputs.to(device)
            outputs = model(data)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            scores = dist
            
            label_score += list(zip(labels.cpu().data.numpy().tolist(), scores.cpu().data.numpy().tolist()))
    
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    # print(labels.shape, scores.shape)
    test_auc = roc_auc_score(labels, scores)
    print('Test set AUC: {:.2f}%'.format(100. * test_auc))
    print('Finished testing.')

            
print('Initializing center c...')
c = init_center_c(train_loader, model, device)
print('Center c initialized.')
for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))
    train(epoch)

test()