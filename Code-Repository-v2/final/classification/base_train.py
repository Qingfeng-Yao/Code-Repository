import random
import torch
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# train/test split
data_test, test_label, data_train, train_label, _, _ = train_and_test_split(imb_data, args, stopwords, label_dict)

tfidf_vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
tfidf_vectorizer.fit(data_train+data_test)
train_features_tfidf = tfidf_vectorizer.transform(data_train) 
test_features_tfidf = tfidf_vectorizer.transform(data_test)

svd = TruncatedSVD(n_components=180)
svd.fit(train_features_tfidf)
train_svd = svd.transform(train_features_tfidf)
test_svd = svd.transform(test_features_tfidf)
scl = preprocessing.StandardScaler()
scl.fit(train_svd)
train_svd_scl = scl.transform(train_svd)
test_svd_scl = scl.transform(test_svd)

# training and tesing
if args.model_name == "nb":
    model = MultinomialNB()
    model.fit(train_features_tfidf,train_label)
    pre = model.predict(test_features_tfidf)

if args.model_name == "lr":
    model = LogisticRegression(C=7)
    # model.fit(train_svd_scl,train_label)
    # pre = model.predict(test_svd_scl)
    model.fit(train_features_tfidf,train_label)
    pre = model.predict(test_features_tfidf)

acc = accuracy_score(test_label, pre)
print("acc:{}".format(acc))
