import csv
import jieba

from sklearn.model_selection import train_test_split

import torch
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import datasets_iterator
from pytorch_pretrained_bert import BertTokenizer
# BertTokenizer reserved tokens: "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"

import dataclass
from . import utils

class Heyspam:
    def __init__(self, root=dataclass.root, is_deep=True, is_jieba=True, is_balanced=True, sequence_length=512): # is_jieba参数为真则使用jieba分词且去停用词，反之则使用bert分字 is_deep参数为真则表示使用深度模型，数据需要转成张量形式 is_balanced参数为真则要求两类数据量相等 sequence_length参数用于限制深度模型的输入序列长度

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = ["普通内容"]
        self.outlier_classes = ["垃圾内容"]
        self.sequence_length = sequence_length

        self.train_set, self.test_set = heyspam_dataset(directory=root, train=True, test=True, is_balanced=is_balanced)
        print("self.train {} self.test {}".format(len(self.train_set), len(self.test_set))) # self.train 48404 self.test 12101  |  self.train 5584 self.test 1396

        for i, row in enumerate(self.train_set):
            if is_deep:
                row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
            else:
                row['label'] = 0 if row['label'] in self.normal_classes else 1
        
        for i, row in enumerate(self.test_set):
            row['label'] = 0 if row['label'] in self.normal_classes else 1

        if not is_jieba:
            self.encoder = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=root+"/bert_cache") 
            self.itos = list(self.encoder.vocab.keys())
            print("bert vocab {}".format(len(self.itos))) # 21128
            self.stoi = {token: index for index, token in enumerate(self.itos)}
            unknown_index = self.stoi['[UNK]']
            pad_index = self.stoi['[PAD]']
            print("bert unknown_index {}".format(unknown_index)) # 100
            print("bert pad_index {}".format(pad_index)) # 0
        else:
            self.stopwords = utils.readStopwords(root+"/stopwords.txt")
            self.itos = []
            for row in datasets_iterator(self.train_set, self.test_set):
                seg_list = jieba.cut(row['text'].strip())
                re_list = []
                for word in seg_list:
                    word = word.strip()
                    if word == "":
                        continue
                    if word in self.stopwords:
                        continue
                    re_list.append(word)
                    self.itos.append(word)
                row['text'] = " ".join(re_list)
            self.itos = list(set(self.itos))
            print("jieba vocab {}".format(len(self.itos))) # 173070
            self.stoi = {token: index for index, token in enumerate(self.itos)}
            unknown_index = len(self.itos)
            pad_index = len(self.itos) + 1
            self.stoi['[UNK]'] = unknown_index
            self.stoi['[PAD]'] = pad_index
            print("jieba unknown_index {}".format(unknown_index))
            print("jieba pad_index {}".format(pad_index)) 

        for row in datasets_iterator(self.train_set, self.test_set):
            if is_jieba:
                if is_deep:
                    vector = [self.stoi.get(token, unknown_index) for token in row['text'].split()[:self.sequence_length]]  
                    row['text'] = torch.LongTensor(vector)
            else:
                text = self.encoder.tokenize(row['text'])
                if is_deep:
                    vector = [self.stoi.get(token, unknown_index) for token in text[:self.sequence_length]]  
                    row['text'] = torch.LongTensor(vector)
                else:
                    row['text'] = " ".join(text)
         
        self.text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        self.label_corpus = [row['label'] for row in datasets_iterator(self.train_set, self.test_set)]
        self.text_train = [row['text'] for row in datasets_iterator(self.train_set)]
        self.label_train = [row['label'] for row in datasets_iterator(self.train_set)]
        self.text_test = [row['text'] for row in datasets_iterator(self.test_set)]
        self.label_test = [row['label'] for row in datasets_iterator(self.test_set)]
        
             

def heyspam_dataset(directory='../data', train=False, test=False, is_balanced=True): # 数据按8:2划分
    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    dict_text2label = {}
    with open("{}/labeled_hupu_docs.csv".format(directory), "r") as csvFile:
        reader = csv.reader(csvFile)
        for item in reader:
            text = item[0].replace('\n', '')
            text = text.strip()
            if text == "":
                continue
            if item[1] not in dict_text2label:
                dict_text2label[item[1]] = [text]
            else:
                dict_text2label[item[1]].append(text)
    print("classes {}".format(len(dict_text2label)))

    train_data = []
    test_data = []
    class_count = []
    for k, v in dict_text2label.items():
        print("{} : {}".format(k, len(v))) # 垃圾内容 : 3490 普通内容 : 57015
        class_count.append(len(v))
    min_class_count = min(class_count)

    for k, v in dict_text2label.items():
        if is_balanced:
            train_x, test_x, train_y, test_y = train_test_split(v[:min_class_count], [k for _ in range(min_class_count)], test_size=0.2, random_state=1)
        else:
            train_x, test_x, train_y, test_y = train_test_split(v, [k for _ in range(len(v))], test_size=0.2, random_state=1)
        for t in train_x:
            train_data.append((t, k)) 
        for t in test_x:
            test_data.append((t, k)) 
    print("train {} test {}".format(len(train_data), len(test_data))) # train 48404 test 12101  |  train 5584 test 1396
    
    for split_set in splits:
        examples = []
        if split_set == 'train':
            data = train_data
        elif split_set == 'test':
            data = test_data

        for t in data:
            text = t[0]
            label = t[1]
            if text:
                examples.append({
                    'text': text,
                    'label': label
                })
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
