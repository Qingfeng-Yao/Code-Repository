import random
import torch
import json
import numpy as np
import csv

from utils import parse_args, readStopwords, train_and_test_split

args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

data_path = 'heybox/input_data/data.json'
with open(data_path, "r") as f:
    imb_data = json.load(f)
stopwords_path = 'static/stopwords.txt'
stopwords = readStopwords(stopwords_path)
label_dict= {"rainbow":0, "daota2":1, "daotabaye":2, "daotazizouqi":3, "monster":4, "zatan":5, "qiusheng":6, "lushi":7, "mingyun2":8, "world":9, "mobile":10, "xianfeng":11, "hardware":12, "union":13, "cloud":14, "zhuji":15, "csgo":16, "pc":17}

data_test, test_label, data_train, train_label, data_val, val_label = train_and_test_split(imb_data, args, stopwords, label_dict)

output_path = "heybox/input_data/text_cnn/three/"

with open(output_path+'train.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_train)):
        tsv_w.writerow([i, train_label[i], data_train[i]]) 

with open(output_path+'test.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_test)):
        tsv_w.writerow([i, test_label[i], data_test[i]]) 

with open(output_path+'dev.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_val)):
        tsv_w.writerow([i, val_label[i], data_val[i]]) 
