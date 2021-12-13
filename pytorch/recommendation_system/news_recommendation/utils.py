import argparse
from nltk.tokenize import word_tokenize
import random
import numpy as np
import jieba 
import math

from text_encoders import MyBertTokenizer

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
    parser.add_argument('--modelname', type=str, default='nrms')

    parser.add_argument('--seed', help='set random seed', type=int, default=123)

    parser.add_argument('--use_ctr', help='whether to use item ctr to get user representation', action="store_true")
    parser.add_argument('--merge_ctr_tar', help='whether to cancat click news content, click news ctr and cand news content to feed additive attention network', action="store_true")
    parser.add_argument('--score_gate', help='whether to user gate to make predictions', action="store_true")
    parser.add_argument('--din', help='whether to use target attention in user encoding', action="store_true")
    parser.add_argument('--cross_atten', help='whether to co-use self-atten and target attention in user attention', action="store_true")
    parser.add_argument('--cross_atten_deep', help='whether to use additive self-atten in cross_atten contidion', action="store_true")

    parser.add_argument('--score_add', help='whether to add self-atten-based score and target-atten-based score', action="store_true")

    parser.add_argument('--add_op', help='whether to add self-atten and target-atten embeds to get user encoding', action="store_true")
    parser.add_argument('--mean_op', help='whether to average self-atten and target-atten embeds to get user encoding', action="store_true")
    parser.add_argument('--max_op', help='whether to max_pool self-atten and target-atten embeds to get user encoding', action="store_true")
    parser.add_argument('--atten_op', help='whether to atten self-atten and target-atten embeds to get user encoding', action="store_true")

    parser.add_argument('--dnn', help='whether to use dnn to get final user representation', action="store_true")
    
    parser.add_argument('--moe', help='whether to use mixture of experts to get final user representation', action="store_true")
    parser.add_argument('--bias', help='whether to use bias net based on moe', action="store_true")
    parser.add_argument('--num_experts', help='number of erperts to use', type=int, default=2)
    parser.add_argument('--mvke', help='whether to use mixture of virtual kernel experts to get final user representation', action="store_true")
    parser.add_argument('--cross_gate', help='whether to use gated cross-selective network to get final user representation', action="store_true")
    parser.add_argument('--dataset', help='path to file: MIND | heybox', type=str, default='MIND')

    parser.add_argument('--word_embed_size', help='word embedding size, if use bert, emb_size=768', type=int, default=300) 
    parser.add_argument('--pretrained_embeddings', help='which pretrained embeddings to use: glove | bert, none is not use', type=str, default='none')

    parser.add_argument('--mimn', help='whether to use mimn to encode user', action="store_true")
    parser.add_argument('--memory_size', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=50) # should <= his_size
    parser.add_argument('--controller_units', type=int, default=50)
    parser.add_argument('--memory_vector_dim', type=int, default=50) # memory_vector_dim=newssize
    parser.add_argument('--read_head_num', type=int, default=1)
    parser.add_argument('--write_head_num', type=int, default=1)
    parser.add_argument('--util_reg', action="store_true")
    parser.add_argument('--mem_induction', action="store_true")
    parser.add_argument('--mask_flag', action="store_true")

    parser.add_argument('--co_attention', help='whether to use co-attention to encode user and news', action="store_true")

    parser.add_argument('--categ_embed_size', help='category embedding size', type=int, default=16) # make news_size(num_heads*head_size+categ_embed_size*2) can divide by num_heads
    parser.add_argument('--epochs', help='max epoch', type=int, default=10)
    parser.add_argument('--neg_number', help='negative samples count', type=int, default=4)
    parser.add_argument('--lr', help='learning_rate', type=float, default=5e-5)
    parser.add_argument('--l2', help='l2 regularization', type=float, default=0.0001)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64) # 不满一个batch的数据不考虑
    parser.add_argument('--eval_batch_size', help='eval batch size', type=int, default=1)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--head_size', type=int, default=16)
    parser.add_argument('--title_size', type=int, default=30)
    parser.add_argument('--his_size', type=int, default=50)
    parser.add_argument('--medialayer', help='middle num units for additive attention network', type=int, default=200)
    parser.add_argument('--max_grad_norm', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='Adamw')
    parser.add_argument('--save', type=int, default=0)

    return parser.parse_args()

def cutWord(text, stopwords=None):
    segs = jieba.cut(text.strip())
    ret = []
    for word in segs:
        word = word.strip()
        if word not in stopwords:
            ret.append(word)
    return ret

def readStopwords(path):
    stopwords = []
    with open(path, "r", encoding="utf-8") as f:
        stopwords = [word.strip() for word in f.readlines()]
    return stopwords

class HeyDataset():
    def __init__(self, args):
        self.title_size = args.title_size
        self.his_size = args.his_size
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.pretrained_embeddings = args.pretrained_embeddings

        train_user_path = 'data/heybox/train/behaviors_train.tsv'
        test_user_path = 'data/heybox/test/behaviors_test.tsv'
        news_path = 'data/heybox/news.tsv'
        # news_ctr_path = 'data/heybox/news_ctr.npz'

        # news_ctr_data = np.load(news_ctr_path, allow_pickle=True)
        # self.news_ctr_origin = news_ctr_data['news_ctr'].tolist()

        with open(news_path, 'r', encoding='utf-8') as f:
            newsfile = f.readlines()
        with open(train_user_path, 'r', encoding='utf-8') as f:
            trainuserfile = f.readlines()
        with open(test_user_path , 'r', encoding='utf-8') as f:
            testuserfile = f.readlines()

        # print(newsfile[0])
        # print(trainuserfile[0])
        # print(testuserfile[0])

        self.news = {}
        num_line = 0
        for line in newsfile:
            num_line += 1
            linesplit = line.split('\t')
            assert len(linesplit)==5, '{}'.format(linesplit)
            if self.pretrained_embeddings == 'bert':
                self.encoder = MyBertTokenizer.from_pretrained('data/ch_bert_cache', cache_dir='data/ch_bert_cache')
                self.news[linesplit[0]] = (linesplit[1], linesplit[2].strip(), self.encoder.encode(linesplit[3].lower()))
            else:
                self.news[linesplit[0]] = (linesplit[1], linesplit[2].strip(), cutWord(linesplit[3].lower(), readStopwords('data/heybox/stopwords.txt')))

        assert num_line == len(self.news)

        self.newsidenx = {'NULL': 0}
        nid = 1
        for id in self.news:
            self.newsidenx[id] = nid
            nid += 1

        # self.news_ctr = {0:{'click':0, 'view':0, 'ctr':0.0, 'ctr_bin': 0}}
        # for news in self.news_ctr_origin.keys():
        #     if news in self.newsidenx:
        #         news_index = self.newsidenx[news]
        #         self.news_ctr[news_index] = self.news_ctr_origin[news]

        self.word_dict = {'PADDING': 0}
        self.categ_dict = {'PADDING': 0}
        self.post_user_dict = {'PADDING': 0}
        self.news_features = [[0] * (self.title_size + 3)]
        self.words = 0
        for newid in self.news:
            title = []
            features = self.news[newid]
            if features[0] not in self.post_user_dict:
                self.post_user_dict[features[0]] = len(self.post_user_dict)
            if features[1] not in self.categ_dict:
                self.categ_dict[features[1]] = len(self.categ_dict)
            
            for w in features[2]:
                if w not in self.word_dict:
                    self.word_dict[w] = len(self.word_dict)
                if self.pretrained_embeddings == 'bert':
                    title.append(w)
                else:
                    title.append(self.word_dict[w])
            self.words += len(title)
            title = title[:self.title_size]
            if self.pretrained_embeddings == 'bert':
                title = title + [self.encoder.stoi['[PAD]']] * (self.title_size - len(title))
            else:
                title = title + [0] * (self.title_size - len(title))
            title.append(self.post_user_dict[features[0]])
            title.append(self.categ_dict[features[1]])
            self.news_features.append(title)

            # if newid not in self.news_ctr_origin:
            #     title.append(0)
            #     self.news_features.append(title)
            #     self.news_ctr_value.append(0.0)
            # else:
            #     title.append(self.news_ctr_origin[newid]['ctr_bin'])
            #     self.news_features.append(title)
            #     self.news_ctr_value.append(self.news_ctr_origin[newid]['ctr'])

        print("num of posts: {}".format(len(self.news)))
        print("ave words in post title: {}".format(self.words/len(self.news)))


        self.negnums = args.neg_number
        self.train_user_his = []
        self.train_candidate = []
        self.train_label = []
        self.train_his_len = []
        self.train_user_id = []
        self.users = {}
        self.clicks = 0
        self.impressions = 0
        self.news_ctr = {0:{'click':0, 'view':0, 'ctr':0.0, 'ctr_bin': 0}}

        for line in trainuserfile:
            self.impressions += 1
            linesplit = line.split('\t')
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

                if self.newsidenx[candidate[0]] not in self.news_ctr:
                    self.news_ctr[self.newsidenx[candidate[0]]] = {'click':0}
                if (candidate[1] == '1'):
                    pnew.append(self.newsidenx[candidate[0]])
                    self.clicks += 1
                    self.news_ctr[self.newsidenx[candidate[0]]]['click'] += 1
                else:
                    nnew.append(self.newsidenx[candidate[0]])

            if len(nnew)==0:
                self.impressions -= 1
                self.clicks -= len(pnew)
                for p in pnew:
                    self.news_ctr[p]['click'] -= 1
                continue

            for pos in pnew:

                if (self.negnums > len(nnew)):
                    negsam = random.sample(nnew * ((self.negnums // len(nnew)) + 1), self.negnums)
                else:
                    negsam = random.sample(nnew, self.negnums)

                negsam.append(pos)

                self.train_candidate.append(negsam)
                self.train_label.append(self.negnums)
                self.train_user_his.append(clickids)
                self.train_his_len.append(click_len)
                self.train_user_id.append(self.users[userid])

        self.eval_candidate = []
        self.eval_label = []
        self.eval_user_his = []
        self.eval_click_len = []
        self.eval_user_id = []

        for line in testuserfile:
            self.impressions += 1
            linesplit = line.split('\t')
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
                if self.newsidenx[candidate[0]] not in self.news_ctr:
                    self.news_ctr[self.newsidenx[candidate[0]]] = {'click':0}
                temp.append(self.newsidenx[candidate[0]])
                temp_label.append(int(candidate[1]))

                if (candidate[1] == '1'):
                    self.clicks += 1
                    self.news_ctr[self.newsidenx[candidate[0]]]['click'] += 1
                    

            if len(temp_label)<2:
                self.impressions -= 1
                if len(temp_label)>0 and temp_label[0] == 1:
                    self.clicks -= 1
                    self.news_ctr[temp[0]]['click'] -= 1
                continue
            
            self.eval_candidate.append(temp)
            self.eval_label.append(temp_label)
            self.eval_user_his.append(clickids)
            self.eval_click_len.append(click_len)
            self.eval_user_id.append(self.users[userid])

        ctr_00 = 0
        ctr_00_01 = 0
        ctr_01_02 = 0
        ctr_02_03 = 0
        ctr_03_04 = 0
        ctr_04_05 = 0
        ctr_05_06 = 0
        ctr_06_07 = 0
        ctr_07_08 = 0
        ctr_08_09 = 0
        ctr_09_10 = 0
        valid_click_num = 0
        for news in self.news_ctr.keys():
            self.news_ctr[news]['view'] = self.impressions
            valid_click_num += self.news_ctr[news]['click']
            self.news_ctr[news]['ctr'] = self.news_ctr[news]['click'] / self.news_ctr[news]['view']

            ctr = self.news_ctr[news]['ctr']
            assert ctr<=1.0
            if ctr == 0.0:
                ctr_00 += 1
                self.news_ctr[news]['ctr_bin'] = 0
            elif ctr > 0.0 and ctr <= 0.1:
                ctr_00_01 += 1
                self.news_ctr[news]['ctr_bin'] = 1
            elif ctr > 0.1 and ctr <= 0.2:
                ctr_01_02 += 1
                self.news_ctr[news]['ctr_bin'] = 2
            elif ctr > 0.2 and ctr <= 0.3:
                ctr_02_03 += 1
                self.news_ctr[news]['ctr_bin'] = 3
            elif ctr > 0.3 and ctr <= 0.4:
                ctr_03_04 += 1
                self.news_ctr[news]['ctr_bin'] = 4
            elif ctr > 0.4 and ctr <= 0.5:
                ctr_04_05 += 1
                self.news_ctr[news]['ctr_bin'] = 5
            elif ctr > 0.5 and ctr <= 0.6:
                ctr_05_06 += 1
                self.news_ctr[news]['ctr_bin'] = 6
            elif ctr > 0.6 and ctr <= 0.7:
                ctr_06_07 += 1
                self.news_ctr[news]['ctr_bin'] = 7
            elif ctr > 0.7 and ctr <= 0.8:
                ctr_07_08 += 1
                self.news_ctr[news]['ctr_bin'] = 8
            elif ctr > 0.8 and ctr <= 0.9:
                ctr_08_09 += 1
                self.news_ctr[news]['ctr_bin'] = 9
            elif ctr > 0.9 and ctr <= 1.0:
                ctr_09_10 += 1
                self.news_ctr[news]['ctr_bin'] = 10
        # print('ctr_00:', ctr_00)
        # print('ctr_00_01:', ctr_00_01)
        # print('ctr_01_02:', ctr_01_02)
        # print('ctr_02_03:', ctr_02_03)
        # print('ctr_03_04:', ctr_03_04)
        # print('ctr_04_05:', ctr_04_05)
        # print('ctr_05_06:', ctr_05_06)
        # print('ctr_06_07:', ctr_06_07)
        # print('ctr_07_08:', ctr_07_08)
        # print('ctr_08_09:', ctr_08_09)
        # print('ctr_09_10:', ctr_09_10)

        assert valid_click_num == self.clicks

        self.news_ctr_value = [0.0,]
        for i, newid in enumerate(self.news.keys()):
            nid = self.newsidenx[newid]
            if nid not in self.news_ctr:
                self.news_features[i+1].append(0)
                self.news_ctr_value.append(0.0)
            else:
                self.news_features[i+1].append(self.news_ctr[nid]['ctr_bin'])
                self.news_ctr_value.append(self.news_ctr[nid]['ctr'])

        self.train_candidate=np.array(self.train_candidate,dtype='int32')
        self.train_label=np.array(self.train_label,dtype='int32')
        self.train_user_his=np.array(self.train_user_his,dtype='int32')
        self.train_his_len = np.array(self.train_his_len, dtype='int32')
        self.train_user_id = np.array(self.train_user_id, dtype='int32')
        self.news_features = np.array(self.news_features)
        self.news_ctr_value = np.array(self.news_ctr_value)


        print("users: {}, impressions: {}, clicks: {}".format(len(self.users), self.impressions, self.clicks))
        print("train samples: {}, test samples: {}".format(len(self.train_candidate), len(self.eval_candidate)))

    def generate_batch_train_data(self):
        idlist = np.arange(len(self.train_label))
        np.random.shuffle(idlist)
        batch_num = len(self.train_label)//self.batch_size
        batches = [idlist[range(self.batch_size*i, min(len(self.train_label),self.batch_size*(i+1)))] for i in range(batch_num)]
        for i in batches:
            item = self.news_features[self.train_candidate[i]] # batch_size, negnums+1, title_size+3
            item_ctr_value = self.news_ctr_value[self.train_candidate[i]] # batch_size, negnums+1,
            user = self.news_features[self.train_user_his[i]] # batch_size, his_size, title_size+3
            user_len = self.train_his_len[i] # batch_size, 
            user_id = self.train_user_id[i] # batch_size, 

            yield (item,item_ctr_value,user,user_len,user_id,self.train_label[i]) # label: batch_size, 


    def generate_batch_eval_data(self):
        for i in range(len(self.eval_candidate)):
            news = [self.news_features[self.eval_candidate[i]]] # 1, num_impression, title_size+3
            news_ctr_value = [self.news_ctr_value[self.eval_candidate[i]]] # 1, num_impression,
            user = [self.news_features[self.eval_user_his[i]]] # 1, his_size, title_size+3
            user_len = [self.eval_click_len[i]] # 1, 
            user_id = [self.eval_user_id[i]] # 1, 
            # test_label = self.eval_label[i]

            yield (news,news_ctr_value,user,user_len,user_id)


class MINDDataset():
    def __init__(self, args):
        self.title_size = args.title_size
        self.his_size = args.his_size
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.pretrained_embeddings = args.pretrained_embeddings
        train_path = 'data/MIND/MINDsmall_train'
        test_path = 'data/MIND/MINDsmall_test'

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
            if self.pretrained_embeddings == 'bert':
                self.encoder = MyBertTokenizer.from_pretrained('data/en_bert_cache', cache_dir='data/en_bert_cache')
                self.news[linesplit[0]] = (linesplit[1].strip(), linesplit[2].strip(), self.encoder.encode(linesplit[3].lower()))
            else:
                self.news[linesplit[0]] = (linesplit[1].strip(), linesplit[2].strip(), word_tokenize(linesplit[3].lower()))

        for line in testnewsfile:
            linesplit = line.split('\t')
            if self.pretrained_embeddings == 'bert':
                self.encoder = MyBertTokenizer.from_pretrained('data/en_bert_cache', cache_dir='data/en_bert_cache')
                self.news[linesplit[0]] = (linesplit[1].strip(), linesplit[2].strip(), self.encoder.encode(linesplit[3].lower()))
            else:
                self.news[linesplit[0]] = (linesplit[1].strip(), linesplit[2].strip(), word_tokenize(linesplit[3].lower()))

        self.newsidenx = {'NULL': 0}
        nid = 1
        for id in self.news:
            self.newsidenx[id] = nid
            nid += 1

        self.word_dict = {'PADDING': 0}
        self.categ_dict = {'PADDING': 0}
        self.news_features = [[0] * (self.title_size + 3)]
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

                if self.pretrained_embeddings == 'bert':
                    title.append(w)
                else:
                    title.append(self.word_dict[w])
                
            self.words += len(title)
            title = title[:self.title_size]
            if self.pretrained_embeddings == 'bert':
                title = title + [self.encoder.stoi['[PAD]']] * (self.title_size - len(title))
            else:
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
        self.train_user_id = []
        self.clicks = 0
        self.impressions = 0
        self.users = {}
        self.news_ctr = {0:{'click':0, 'view':0, 'ctr':0.0, 'ctr_bin': 0}}

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
                if self.newsidenx[candidate[0]] not in self.news_ctr:
                    self.news_ctr[self.newsidenx[candidate[0]]] = {'click':0}
                if (candidate[1] == '1'):
                    pnew.append(self.newsidenx[candidate[0]])
                    self.clicks += 1
                    self.news_ctr[self.newsidenx[candidate[0]]]['click'] += 1
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
                self.train_user_id.append(self.users[userid])

        self.eval_candidate = []
        self.eval_label = []
        self.eval_user_his = []
        self.eval_click_len = []
        self.eval_user_id = []

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
                if self.newsidenx[candidate[0]] not in self.news_ctr:
                    self.news_ctr[self.newsidenx[candidate[0]]] = {'click':0}

                temp.append(self.newsidenx[candidate[0]])
                temp_label.append(int(candidate[1]))
                if (candidate[1] == '1'):
                    self.clicks += 1
                    self.news_ctr[self.newsidenx[candidate[0]]]['click'] += 1

            self.eval_candidate.append(temp)
            self.eval_label.append(temp_label)
            self.eval_user_his.append(clickids)
            self.eval_click_len.append(click_len)
            self.eval_user_id.append(self.users[userid])

        ctr_00 = 0
        ctr_00_01 = 0
        ctr_01_02 = 0
        ctr_02_03 = 0
        ctr_03_04 = 0
        ctr_04_05 = 0
        ctr_05_06 = 0
        ctr_06_07 = 0
        ctr_07_08 = 0
        ctr_08_09 = 0
        ctr_09_10 = 0
        valid_click_num = 0
        for news in self.news_ctr.keys():
            self.news_ctr[news]['view'] = self.impressions
            valid_click_num += self.news_ctr[news]['click']
            self.news_ctr[news]['ctr'] = self.news_ctr[news]['click'] / self.news_ctr[news]['view']

            ctr = self.news_ctr[news]['ctr']
            assert ctr<=1.0
            if ctr == 0.0:
                ctr_00 += 1
                self.news_ctr[news]['ctr_bin'] = 0
            elif ctr > 0.0 and ctr <= 0.1:
                ctr_00_01 += 1
                self.news_ctr[news]['ctr_bin'] = 1
            elif ctr > 0.1 and ctr <= 0.2:
                ctr_01_02 += 1
                self.news_ctr[news]['ctr_bin'] = 2
            elif ctr > 0.2 and ctr <= 0.3:
                ctr_02_03 += 1
                self.news_ctr[news]['ctr_bin'] = 3
            elif ctr > 0.3 and ctr <= 0.4:
                ctr_03_04 += 1
                self.news_ctr[news]['ctr_bin'] = 4
            elif ctr > 0.4 and ctr <= 0.5:
                ctr_04_05 += 1
                self.news_ctr[news]['ctr_bin'] = 5
            elif ctr > 0.5 and ctr <= 0.6:
                ctr_05_06 += 1
                self.news_ctr[news]['ctr_bin'] = 6
            elif ctr > 0.6 and ctr <= 0.7:
                ctr_06_07 += 1
                self.news_ctr[news]['ctr_bin'] = 7
            elif ctr > 0.7 and ctr <= 0.8:
                ctr_07_08 += 1
                self.news_ctr[news]['ctr_bin'] = 8
            elif ctr > 0.8 and ctr <= 0.9:
                ctr_08_09 += 1
                self.news_ctr[news]['ctr_bin'] = 9
            elif ctr > 0.9 and ctr <= 1.0:
                ctr_09_10 += 1
                self.news_ctr[news]['ctr_bin'] = 10
        # print('ctr_00:', ctr_00)
        # print('ctr_00_01:', ctr_00_01)
        # print('ctr_01_02:', ctr_01_02)
        # print('ctr_02_03:', ctr_02_03)
        # print('ctr_03_04:', ctr_03_04)
        # print('ctr_04_05:', ctr_04_05)
        # print('ctr_05_06:', ctr_05_06)
        # print('ctr_06_07:', ctr_06_07)
        # print('ctr_07_08:', ctr_07_08)
        # print('ctr_08_09:', ctr_08_09)
        # print('ctr_09_10:', ctr_09_10)

        assert valid_click_num == self.clicks

        self.news_ctr_value = [0.0,]
        for i, newid in enumerate(self.news.keys()):
            nid = self.newsidenx[newid]
            if nid not in self.news_ctr:
                self.news_features[i+1].append(0)
                self.news_ctr_value.append(0.0)
            else:
                self.news_features[i+1].append(self.news_ctr[nid]['ctr_bin'])
                self.news_ctr_value.append(self.news_ctr[nid]['ctr'])

        self.train_candidate=np.array(self.train_candidate,dtype='int32')
        self.train_label=np.array(self.train_label,dtype='int32')
        self.train_user_his=np.array(self.train_user_his,dtype='int32')
        self.train_his_len = np.array(self.train_his_len, dtype='int32')
        self.train_user_id = np.array(self.train_user_id, dtype='int32')
        self.news_features = np.array(self.news_features)
        self.news_ctr_value = np.array(self.news_ctr_value)

        print("users: {}, impressions: {}, clicks: {}".format(len(self.users), self.impressions, self.clicks))
        print("train samples: {}, test samples: {}".format(len(self.train_candidate), len(self.eval_candidate)))

    def generate_batch_train_data(self):
        idlist = np.arange(len(self.train_label))
        np.random.shuffle(idlist)
        batch_num = len(self.train_label)//self.batch_size
        batches = [idlist[range(self.batch_size*i, min(len(self.train_label),self.batch_size*(i+1)))] for i in range(batch_num)]
        for i in batches:
            item = self.news_features[self.train_candidate[i]] # batch_size, negnums+1, title_size+3
            item_ctr_value = self.news_ctr_value[self.train_candidate[i]] # batch_size, negnums+1,
            user = self.news_features[self.train_user_his[i]] # batch_size, his_size, title_size+3
            user_len = self.train_his_len[i] # batch_size, 
            user_id = self.train_user_id[i] # batch_size, 

            yield (item,item_ctr_value,user,user_len,user_id,self.train_label[i]) # label: batch_size, 


    def generate_batch_eval_data(self):
        for i in range(len(self.eval_candidate)):
            news = [self.news_features[self.eval_candidate[i]]] # 1, num_impression, title_size+3
            news_ctr_value = [self.news_ctr_value[self.eval_candidate[i]]] # 1, num_impression,
            user = [self.news_features[self.eval_user_his[i]]] # 1, his_size, title_size+3
            user_len = [self.eval_click_len[i]] # 1, 
            user_id = [self.eval_user_id[i]] # 1, 
            # test_label = self.eval_label[i]

            yield (news,news_ctr_value,user,user_len,user_id)
