import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import time

import torch
import torch.optim as optim
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchnlp.word_to_vector import GloVe
from torchtext.vocab import FastText

import datasets
from cvdd_models import *
import util

## 参数设置
parser = argparse.ArgumentParser(description='pytorch text anomaly detection(one-class) using cvdd model')
parser.add_argument(
    '--dataset',
    type=str, 
    default='reuters',
    help='reuters | newsgroup')
parser.add_argument(
    '--normal_class', 
    type=int, 
    default=0,
    help='specify the normal class of the dataset (all other classes are considered anomalous).')
parser.add_argument(
    '--cuda-device',
    type=str, 
    default='cuda:0',
    help='cuda:0 | ...')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disable cuda training')
parser.add_argument(
    '--lr_milestones', nargs="+", type=int, default=[-1], help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed')

parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    help='input batch size for training')
parser.add_argument(
    '--embedding_size',
    type=int,
    default=None,
    help='size of the word vector embedding')
parser.add_argument(
    '--pretrained_model',
    default='FastText_en',
    help='FastText_en | GloVe_6B | GloVe_42B | GloVe_840B | GloVe_twitter.27B')
parser.add_argument(
    '--ad_score',
    type=str, 
    default='context_dist_mean',
    help='context_dist_mean | context_best')
parser.add_argument(
    '--n_attention_heads',
    type=int,
    default=3,
    help='number of attention heads in self-attention module')
parser.add_argument(
    '--attention_size',
    type=int,
    default=100,
    help='self-attention module dimensionality')

parser.add_argument(
    '--lr', type=float, default=0.01, help='initial learning rate for training')
parser.add_argument(
    '--weight_decay', type=float, default=0.5e-6, help='weight decay (L2 penalty) hyperparameter')
parser.add_argument(
    '--n_epochs',
    type=int,
    default=100,
    help='number of epochs to train')
parser.add_argument(
    '--lambda_p', 
    type=float, 
    default=1.0,
    help='hyperparameter for context vector orthogonality regularization P = (CCT - I)')
parser.add_argument(
    '--alpha_scheduler',
    type=str, 
    default='logarithmic',
    help='logarithmic | soft | linear | hard')

parser.add_argument(
    '--n_jobs_dataloader',
    type=int,
    default=0,
    help='number of workers for data loading. 0 means that the data will be loaded in the main process')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.cuda_device if args.cuda else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# alpha annealing strategy
args.alpha_milestones = np.arange(1, 6) * int(args.n_epochs / 5)  # 5 equidistant milestones over n_epochs
if args.alpha_scheduler == 'soft':
    args.alphas = [0.0] * 5
if args.alpha_scheduler == 'linear':
    args.alphas = np.linspace(.2, 1, 5)
if args.alpha_scheduler == 'logarithmic':
    args.alphas = np.logspace(-4, 0, 5)
if args.alpha_scheduler == 'hard':
    args.alphas = [100.0] * 4

print('Args:', args)

## 数据下载
dataset = getattr(datasets, "OC_DATA")(args.dataset, args.normal_class)

def collate_fn(batch):
    """ list of tensors to a batch tensors """
    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t().contiguous())

    text_batch, _ = stack_and_pad_tensors([row['text'] for row in batch])
    label_batch = torch.stack([row['label'] for row in batch])

    return transpose(text_batch), label_batch.float()

train_sampler = BucketBatchSampler(dataset.train_set, batch_size=args.batch_size, drop_last=True,
                                           sort_key=lambda r: len(r['text']))
test_sampler = BucketBatchSampler(dataset.test_set, batch_size=args.batch_size, drop_last=True,
                                          sort_key=lambda r: len(r['text']))

train_loader = torch.utils.data.DataLoader(
    dataset=dataset.train_set, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=args.n_jobs_dataloader)

test_loader = torch.utils.data.DataLoader(
    dataset=dataset.test_set,
    batch_sampler=test_sampler,
    collate_fn=collate_fn, num_workers=args.n_jobs_dataloader)

## 模型及优化器
if args.pretrained_model == 'GloVe_6B':
    assert args.embedding_size in (50, 100, 200, 300)
    word_vectors = GloVe(name='6B', dim=args.embedding_size, cache='data/word_vectors_cache')
if args.pretrained_model == 'GloVe_42B':
    assert args.embedding_size == 300
    word_vectors = GloVe(name='42B', cache='data/word_vectors_cache')
if args.pretrained_model == 'GloVe_840B':
    assert args.embedding_size == 300
    word_vectors = GloVe(name='840B', cache='data/word_vectors_cache')
if args.pretrained_model == 'GloVe_twitter.27B':
    assert args.embedding_size in (25, 50, 100, 200)
    word_vectors = GloVe(name='twitter.27B', dim=args.embedding_size, cache='data/word_vectors_cache')
if args.pretrained_model == 'FastText_en':
    assert args.embedding_size == 300
    word_vectors = FastText(language='en', cache='data/word_vectors_cache')

embedding = MyEmbedding(dataset.encoder.vocab_size, args.embedding_size, update_embedding=False, reduction='none', normalize=False)
# Init embedding with pre-trained word vectors
for i, token in enumerate(dataset.encoder.vocab):
    embedding.weight.data[i] = word_vectors[token]

model = CVDDNet(embedding, attention_size=args.attention_size, n_attention_heads=args.n_attention_heads)
model = model.to(device)

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

pytorch_total_params, pytorch_params = sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())
print('pytorch_total_params {} pytorch_params {}'.format(pytorch_total_params, pytorch_params))

## 训练及测试
def train(epoch):
    
    if epoch in args.lr_milestones:
        print('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

    if epoch in args.alpha_milestones:
        global alpha_i
        model.alpha = float(args.alphas[alpha_i])
        print('  Temperature alpha scheduler: new alpha is %g' % model.alpha)
        alpha_i += 1

    epoch_loss = 0.0
    n_batches = 0
    epoch_start_time = time.time()
    for data in train_loader:
        text_batch, _  = data
        text_batch = text_batch.to(device)
        # text_batch.shape = (sentence_length, batch_size)

        # Zero the network parameter gradients
        optimizer.zero_grad()

        # Update network parameters via backpropagation: forward + backward + optimize

        # forward pass
        cosine_dists, context_weights, A = model(text_batch)
        scores = context_weights * cosine_dists
        # scores.shape = (batch_size, n_attention_heads)
        # A.shape = (batch_size, n_attention_heads, sentence_length)

        # get orthogonality penalty: P = (CCT - I)
        I = torch.eye(args.n_attention_heads).to(device)
        CCT = model.c @ model.c.transpose(1, 2)
        P = torch.mean((CCT.squeeze() - I) ** 2)

        # compute loss
        loss_P = args.lambda_p * P
        loss_emp = torch.mean(torch.sum(scores, dim=1))
        loss = loss_emp + loss_P

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # clip gradient norms in [-0.5, 0.5]
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1
    scheduler.step()

    epoch_train_time = time.time() - epoch_start_time
    print(f'| Epoch: {epoch + 1:03}/{args.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                f'| Train Loss: {epoch_loss / n_batches:.6f} |')


def test():
    print('Starting testing...')
    epoch_loss = 0.0
    n_batches = 0
    dists_per_head = ()
    label_score_head = []
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            text_batch, label_batch = data
            text_batch, label_batch = text_batch.to(device), label_batch.to(device)

            # forward pass
            cosine_dists, context_weights, A = model(text_batch)
            scores = context_weights * cosine_dists
            _, best_att_head = torch.min(scores, dim=1)

            # get orthogonality penalty: P = (CCT - I)
            I = torch.eye(args.n_attention_heads).to(device)
            CCT = model.c @ model.c.transpose(1, 2)
            P = torch.mean((CCT.squeeze() - I) ** 2)

            # compute loss
            loss_P = args.lambda_p * P
            loss_emp = torch.mean(torch.sum(scores, dim=1))
            loss = loss_emp + loss_P

            # Save tuples of (idx, label, score, best_att_head) in a list
            dists_per_head += (cosine_dists.cpu().data.numpy(),)
            ad_scores = torch.mean(cosine_dists, dim=1)
            label_score_head += list(zip(label_batch.cpu().data.numpy().tolist(),
                                                ad_scores.cpu().data.numpy().tolist(),
                                                best_att_head.cpu().data.numpy().tolist()))

            epoch_loss += loss.item()
            n_batches += 1

    test_time = time.time() - start_time
    test_dists = np.concatenate(dists_per_head)
    test_scores = label_score_head
    # Compute AUC
    labels, scores, _ = zip(*label_score_head)
    labels = np.array(labels)
    scores = np.array(scores)

    if np.sum(labels) > 0:
        best_context = None
        if args.ad_score == 'context_dist_mean':
            test_auc = roc_auc_score(labels, scores)
        if args.ad_score == 'context_best':
            test_auc = 0.0
            for context in range(args.n_attention_heads):
                auc_candidate = roc_auc_score(labels, test_dists[:, context])
                if auc_candidate > test_auc:
                    test_auc = auc_candidate
                    best_context = context
                else:
                    pass
    else:
        best_context = None
        test_auc = 0.0

    print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
    print('Test AUC: {:.2f}%'.format(100. * test_auc))
    print(f'Test Best Context: {best_context}')
    print('Test Time: {:.3f}s'.format(test_time))
    print('Finished testing.')



model.c.data = torch.from_numpy(
            util.initialize_context_vectors(model, train_loader, device)[np.newaxis, :]).to(device)  

print('Starting training...')
start_time = time.time()
model.train()
alpha_i = 0
for epoch in range(args.n_epochs):
    train(epoch)
train_time = time.time() - start_time
print('Training Time: {:.3f}s'.format(train_time))
print('Finished training.')
    
test()

    
