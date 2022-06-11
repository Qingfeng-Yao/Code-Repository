import csv
import jieba

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import datasets_iterator
from pytorch_pretrained_bert import BertTokenizer
# BertTokenizer reserved tokens: "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"

import datasets

class Heyspam:
    def __init__(self, normal_class=0, root=datasets.root, is_deep=True, is_jieba=False): # normal为-1则训练使用全部类

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = ["普通内容"]
        self.outlier_classes = ["垃圾内容"]

        self.train_set, self.test_set = heyspam_dataset(directory=root, train=True, test=True, special=True)

        if normal_class == -1:
            for i, row in enumerate(self.train_set):
                if is_deep:
                    row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
                else:
                    row['label'] = 0 if row['label'] in self.normal_classes else 1
        else:
            train_idx_normal = []  # for subsetting train_set to normal class
            for i, row in enumerate(self.train_set):
                if row['label'] in self.normal_classes:
                    train_idx_normal.append(i)
                    row['label'] = torch.tensor(0)
                else:
                    row['label'] = torch.tensor(1)

            # Subset train_set to normal class
            self.train_set = Subset(self.train_set, train_idx_normal)

        for i, row in enumerate(self.test_set):
            if is_deep:
                row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
            else:
                row['label'] = 0 if row['label'] in self.normal_classes else 1


        print("self.train {} self.test {}".format(len(self.train_set), len(self.test_set)))

        self.encoder = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=root+"/bert_cache") 
        self.itos = list(self.encoder.vocab.keys())
        print("vocab {}".format(len(self.itos))) 
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        unknown_index = self.stoi['[UNK]']
        print("unknown_index {}".format(unknown_index)) # 100

        for row in datasets_iterator(self.train_set, self.test_set):
            if is_jieba:
                seg_list = jieba.cut(row['text'].strip())
                row['text'] = " ".join(seg_list)
            else:
                text = self.encoder.tokenize(row['text'])
                if is_deep:
                    vector = [self.stoi.get(token, unknown_index) for token in text[:512]]  
                    row['text'] = torch.LongTensor(vector)
                else:
                    row['text'] = " ".join(text)
         
        self.text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        self.label_corpus = [row['label'] for row in datasets_iterator(self.train_set, self.test_set)]
        self.text_train = [row['text'] for row in datasets_iterator(self.train_set)]
        self.label_train = [row['label'] for row in datasets_iterator(self.train_set)]
        self.text_test = [row['text'] for row in datasets_iterator(self.test_set)]
        self.label_test = [row['label'] for row in datasets_iterator(self.test_set)]
        
             

def heyspam_dataset(directory='../data', train=False, test=False, special=False): # 参数special为真则需要验证特例

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    dict_text2label = {}
    csvFile = open("{}/heyspam/spam_2024.csv".format(directory), "r")
    reader = csv.reader(csvFile)
    for item in reader:
        text = item[0].replace('\n', ' ')
        text = text.strip()
        if text == "":
            continue
        if item[1] not in dict_text2label:
            dict_text2label[item[1]] = [text]
        else:
            dict_text2label[item[1]].append(text)
    csvFile.close()
    print("classes {}".format(len(dict_text2label)))
    train_data = []
    test_data = []
    for k, v in dict_text2label.items():
        print("{} : {}".format(k, len(v)))
        train_x, test_x, train_y, test_y = train_test_split(v, [k for _ in range(len(v))], test_size=0.1, random_state=1)
        for t in train_x:
            train_data.append((t, k)) 
        for t in test_x:
            test_data.append((t, k)) 
    print("train {} test {}".format(len(train_data), len(test_data)))
    if special:
        examples = []
        for t in train_data:
            text = t[0]
            label = t[1]
            if text:
                examples.append({
                    'text': text,
                    'label': label
                })
        for t in test_data:
            text = t[0]
            label = t[1]
            if text:
                examples.append({
                    'text': text,
                    'label': label
                })
        ret.append(Dataset(examples))  
        examples = [] 
        with open("{}/heyspam/spam_special.txt".format(directory), "r") as spam_examples:
            for e in spam_examples.readlines():
                examples.append({
                    'text': e.strip(),
                    'label': "垃圾内容"
                })
        with open("{}/heyspam/normal_special.txt".format(directory), "r") as normal_examples:
            for e in normal_examples.readlines():
                examples.append({
                    'text': e.strip(),
                    'label': "普通内容"
                })
        print("special {}".format(len(examples)))
        ret.append(Dataset(examples))
 
    else:
        for split_set in splits:
            examples = []
            if split_set == 'train':
                for t in train_data:
                    text = t[0]
                    label = t[1]
                    if text:
                        examples.append({
                            'text': text,
                            'label': label
                        })
            if split_set == 'test':
                for t in test_data:
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
