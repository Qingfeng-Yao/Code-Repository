import argparse
from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
from torchnlp.word_to_vector import GloVe
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors

import datasets
from model import *
  
## Training settings
parser = argparse.ArgumentParser(description='PyTorch text anomaly detection using cvdd')
parser.add_argument(
    '--dataset',
    default='Reuters',
    help='Reuters | Newsgroups20 | Heyspam')
parser.add_argument(
    '--model', default='cvdd_net', help='cvdd_net')
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
    default=64,
    help='input batch size for training')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=64,
    help='input batch size for testing')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train')
parser.add_argument(
    '--lr', type=float, default=0.01, help='learning rate')
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

if args.dataset == 'Heyspam':
    embedding = BERT()
else:
    word_vectors = GloVe(name='6B', dim=300, cache="data/word_vectors_cache")

    embedding = MyEmbedding(dataset.encoder.vocab_size, 300, update_embedding=False, embedding_reduction='none', normalize_embedding=False)
    ### Init embedding with pre-trained word vectors
    for i, token in enumerate(dataset.encoder.vocab):
        embedding.weight.data[i] = word_vectors[token]

### stack expects each tensor to be equal size

def collate_fn(batch):
    """ list of tensors to a batch tensors """
    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    text_batch, _ = stack_and_pad_tensors([row['text'] for row in batch])
    label_batch = torch.stack([row['label'] for row in batch])

    return transpose(text_batch), label_batch.float()

train_sampler = BucketBatchSampler(dataset.train_set, batch_size=args.batch_size, drop_last=False,
                                           sort_key=lambda r: len(r['text']))
test_sampler = BucketBatchSampler(dataset.test_set, batch_size=args.test_batch_size, drop_last=False,
                                    sort_key=lambda r: len(r['text']))

train_loader = torch.utils.data.DataLoader(dataset=dataset.train_set, batch_sampler=train_sampler, collate_fn=collate_fn, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset=dataset.test_set, batch_sampler=test_sampler, collate_fn=collate_fn, **kwargs)

## Model settings
if args.model == 'cvdd_net':
    model = CVDDNet(embedding, attention_size=150, n_attention_heads=3)
 
model.to(device)
### Initialize context vectors
model.c.data = torch.from_numpy(initialize_context_vectors(model, train_loader, device)[np.newaxis, :]).to(device)

## Optimizer
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=0.5e-6)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=(40,80), gamma=0.1)

## Training and testing
def train(epoch):
    model.train()
    train_loss = 0

    if epoch in (40,80):
        print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

    if epoch in alpha_milestones:
        global alpha_i
        model.alpha = float(alphas[alpha_i])
        print('  Temperature alpha scheduler: new alpha is %g' % model.alpha)
        alpha_i += 1

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        text_batch, _ = data
        text_batch = text_batch.to(device)
        optimizer.zero_grad()
        cosine_dists, context_weights, A = model(text_batch)
        scores = context_weights * cosine_dists

        I = torch.eye(model.n_attention_heads).to(device)
        CCT = model.c @ model.c.transpose(1, 2)
        P = torch.mean((CCT.squeeze() - I) ** 2)

        loss_P = 1.0 * P
        loss_emp = torch.mean(torch.sum(scores, dim=1))
        loss = loss_emp + loss_P
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # clip gradient norms in [-0.5, 0.5]
        optimizer.step()

        train_loss += loss.item()

        pbar.update(text_batch.size(0))
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
            data, labels = inputs.to(device), labels.to(device) 
            cosine_dists, context_weights, A = model(data)
            scores = context_weights * cosine_dists

            ad_scores = torch.mean(cosine_dists, dim=1)
            label_score += list(zip(labels.cpu().data.numpy().tolist(), ad_scores.cpu().data.numpy().tolist()))
    
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    print('Test set AUC: {:.2f}%'.format(100. * test_auc))
    print('Finished testing.')

alpha_milestones = np.arange(1, 6) * int(args.epochs / 5)
alphas = np.logspace(-4, 0, 5)
alpha_i = 0
for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))
    train(epoch)

test()