import argparse
from nltk.tokenize import word_tokenize
import random
import numpy as np

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='set gpu device number 0-3', type=str, default='cuda:0')
    parser.add_argument('--use_multi_gpu', help='whether to use multi gpus', action="store_true")
    parser.add_argument('--foldname', type=str, default='nrms')
    parser.add_argument('--dataset', help='path to file', type=str, default='small')

    parser.add_argument('--word_embed_size', help='word embedding size', type=int, default=300)
    parser.add_argument('--use_pretrained_embeddings', help='whether to use pretrained embeddings', action="store_false")
    parser.add_argument('--categ_embed_size', help='category embedding size', type=int, default=16) # make news_size(num_heads*head_size+categ_embed_size*2) can divide by num_heads
    parser.add_argument('--epochs', help='max epoch', type=int, default=7)
    parser.add_argument('--neg_number', help='negative samples count', type=int, default=4)
    parser.add_argument('-lr', help='learning_rate', type=float, default=0.00005)
    parser.add_argument('-l2', help='l2 regularization', type=float, default=0.0001)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--head_size', type=int, default=16)
    parser.add_argument('--title_size', type=int, default=30)
    parser.add_argument('--his_size', type=int, default=50)
    parser.add_argument('--medialayer', help='middle num units for title layer', type=int, default=200)
    parser.add_argument('--max_grad_norm', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='Adamw')
    parser.add_argument('--save', type=int, default=0)

    return parser.parse_args()

class MINDDataset():
    def __init__(self, args):
        self.title_size = args.title_size
        self.his_size = args.his_size
        self.batch_size = args.batch_size
        if args.dataset == 'small':
            train_path = 'data/MINDsmall_train'
            test_path = 'data/MINDsmall_test'
        else:
            raise NotImplementedError


        news_path = '/news.tsv'
        user_path = '/behaviors.tsv'
        with open(train_path + news_path, 'r', encoding='utf-8') as f:
            trainnewsfile = f.readlines()
        with open(train_path + user_path, 'r', encoding='utf-8') as f:
            trainuserfile = f.readlines()
        with open(test_path + news_path, 'r', encoding='utf-8') as f:
            testnewsfile = f.readlines()
        with open(test_path + user_path, 'r', encoding='utf-8') as f:
            testuserfile = f.readlines()

        # print(trainnewsfile[0])
        # print(trainuserfile[0])
        # print(testnewsfile[0])
        # print(testuserfile[0])

        self.news = {}
        for line in trainnewsfile:
            linesplit = line.split('\t')
            self.news[linesplit[0]] = (linesplit[1].strip(), linesplit[2].strip(), word_tokenize(linesplit[3].lower()))

        for line in testnewsfile:
            linesplit = line.split('\t')
            self.news[linesplit[0]] = (linesplit[1].strip(), linesplit[2].strip(), word_tokenize(linesplit[3].lower()))

        self.newsidenx = {'NULL': 0}
        nid = 1
        for id in self.news:
            self.newsidenx[id] = nid
            nid += 1

        self.word_dict = {'PADDING': 0}
        self.categ_dict = {'PADDING': 0}
        self.news_features = [[0] * (self.title_size + 2)]
        self.words = 0
        for newid in self.news:
            title = []
            features = self.news[newid]
            if features[0] not in self.categ_dict:
                self.categ_dict[features[0]] = len(self.categ_dict)
            if features[1] not in self.categ_dict:
                self.categ_dict[features[1]] = len(self.categ_dict)
            for w in features[2]:
                if w not in self.word_dict:
                    self.word_dict[w] = len(self.word_dict)
                title.append(self.word_dict[w])
            self.words += len(title)
            title = title[:self.title_size]
            title = title + [0] * (self.title_size - len(title))
            title.append(self.categ_dict[features[0]])
            title.append(self.categ_dict[features[1]])
            self.news_features.append(title)

        print("num of news: {}".format(len(self.news)))
        print("ave words in news title: {}".format(self.words/len(self.news)))

        self.negnums = args.neg_number
        self.train_user_his = []
        self.train_candidate = []
        self.train_label = []
        self.train_his_len = []
        self.clicks = 0
        self.impressions = 0
        self.users = {}

        for line in trainuserfile:
            linesplit = line.split('\t')
            self.impressions += 1
            userid = linesplit[1].strip()
            if userid not in self.users:
                self.users[userid] = len(self.users)

            clickids = [n for n in linesplit[3].split(' ') if n != '']
            clickids = clickids[-self.his_size:]
            click_len = len(clickids)
            clickids = clickids + ['NULL'] * (self.his_size - len(clickids))
            clickids = [self.newsidenx[n] for n in clickids]

            pnew = []
            nnew = []
            for candidate in linesplit[4].split(' '):
                candidate = candidate.strip().split('-')
                if (candidate[1] == '1'):
                    pnew.append(self.newsidenx[candidate[0]])
                    self.clicks += 1
                else:
                    nnew.append(self.newsidenx[candidate[0]])

            for pos in pnew:

                if (self.negnums > len(nnew)):
                    negsam = random.sample(nnew * ((self.negnums // len(nnew)) + 1), self.negnums)
                else:
                    negsam = random.sample(nnew, self.negnums)

                negsam.append(pos)

                # shuffle
                self.train_candidate.append(negsam)
                self.train_label.append(self.negnums)
                self.train_user_his.append(clickids)
                self.train_his_len.append(click_len)

        self.eval_candidate = []
        self.eval_label = []
        self.eval_user_his = []
        self.eval_click_len = []

        for line in testuserfile:
            linesplit = line.split('\t')
            self.impressions += 1
            userid = linesplit[1].strip()
            if userid not in self.users:
                self.users[userid] = len(self.users)

            clickids = [n for n in linesplit[3].split(' ') if n != '']
            clickids = clickids[-self.his_size:]
            click_len = len(clickids)
            clickids = clickids + ['NULL'] * (self.his_size - len(clickids))
            clickids = [self.newsidenx[n] for n in clickids]

            temp = []
            temp_label = []
            for candidate in linesplit[4].split(' '):
                candidate = candidate.strip().split('-')
                temp.append(self.newsidenx[candidate[0]])
                temp_label.append(int(candidate[1]))
                if (candidate[1] == '1'):
                    self.clicks += 1
            
            self.eval_candidate.append(temp)
            self.eval_label.append(temp_label)
            self.eval_user_his.append(clickids)
            self.eval_click_len.append(click_len)

        self.train_candidate=np.array(self.train_candidate,dtype='int32')
        self.train_label=np.array(self.train_label,dtype='int32')
        self.train_user_his=np.array(self.train_user_his,dtype='int32')
        self.train_his_len = np.array(self.train_his_len, dtype='int32')
        self.news_features = np.array(self.news_features)

        print("users: {}, impressions: {}, clicks: {}".format(len(self.users), self.impressions, self.clicks))

    def generate_batch_train_data(self):
        idlist = np.arange(len(self.train_label))
        np.random.shuffle(idlist)
        batches = [idlist[range(self.batch_size*i, min(len(self.train_label),self.batch_size*(i+1)))] for i in range(len(self.train_label)//self.batch_size+1)]
        for i in batches:
            item = self.news_features[self.train_candidate[i]] # batch_size, negnums+1, title_size+2
            user = self.news_features[self.train_user_his[i]] # batch_size, his_size, title_size+2
            user_len = self.train_his_len[i] # batch_size, 

            yield (item,user,user_len,self.train_label[i]) # label: batch_size, 


    def generate_batch_eval_data(self):
        for i in range(len(self.eval_candidate)):
            news = [self.news_features[self.eval_candidate[i]]] # 1, num_impression, title_size+2
            user = [self.news_features[self.eval_user_his[i]]] # 1, his_size, title_size+2
            user_len = [self.eval_click_len[i]] # 1, 
            # test_label = self.eval_label[i]

            yield (news,user,user_len)
