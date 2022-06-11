import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchnlp.encoders.text.text_encoder import pad_tensor

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score

import dataclass
import models

parser = argparse.ArgumentParser(description='using supervised deep learning')
parser.add_argument(
    '--dataset',
    default='Heyspam',
    help='Heyspam')
parser.add_argument(
    '--model', default='textcnn', help='textcnn | lstm')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed')
parser.add_argument('--embedding_size', type=int, default=100)
parser.add_argument('--sequence_length', type=int, default=256) 

parser.add_argument('--n_hidden', type=int, default=16) # lstm
parser.add_argument('--filter_sizes', default=[3, 4, 5]) # textcnn
parser.add_argument('--num_filters', type=int, default=16) # textcnn

parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train')
parser.add_argument(
    '--lr', type=float, default=0.001, help='learning rate')

parser.add_argument(
    '--no-jieba',
    action='store_true',
    default=False)
parser.add_argument(
    '--no-balanced',
    action='store_true',
    default=False)


args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset = getattr(dataclass, args.dataset)(is_deep=True, is_jieba=not args.no_jieba, is_balanced=not args.no_balanced, sequence_length=args.sequence_length)

train_padded = [pad_tensor(tensor, args.sequence_length, dataset.stoi['[PAD]']) for tensor in dataset.text_train]
train_padded = torch.stack(train_padded, dim=0).contiguous()
train_labels = torch.stack(dataset.label_train)
train_padded = train_padded.to(device)
train_labels = train_labels.to(device)

if args.model == "textcnn":
    model = models.TextCNN(args, vocab_size=len(dataset.stoi), num_classes=dataset.n_classes)  
elif args.model == "lstm":
    model = models.BiLSTM_Attention(args, vocab_size=len(dataset.stoi), num_classes=dataset.n_classes, device=device)
    
model.to(device)
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Training
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_padded)

    # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, train_labels)
    pbar = tqdm(total=len(train_padded))
    pbar.update(train_padded.size(0))
    pbar.set_description('Epoch: {} |  cost ={:.6f}'.format(epoch + 1, loss))
    pbar.close()

    loss.backward()
    optimizer.step()

# Test
test_padded = [pad_tensor(tensor, args.sequence_length, dataset.stoi['[PAD]']) for tensor in dataset.text_test]
test_padded = torch.stack(test_padded, dim=0).contiguous()
test_padded = test_padded.to(device)

# Predict
model.eval()
with torch.no_grad():
    pre = model(test_padded).data.max(1)[1]
# print(pre.shape) # torch.Size([1396])
 
prec, recall, f1, _ = precision_recall_fscore_support(dataset.label_test, pre.data.cpu().numpy(), average="weighted")  
acc = accuracy_score(dataset.label_test, pre.data.cpu().numpy())
auc = roc_auc_score(dataset.label_test, pre.data.cpu().numpy())
print("prec:{} ; reacll:{} ; f1:{} ; acc:{} ; auc:{}".format(prec, recall, f1, acc, auc))
