import csv
import xlrd

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score

from spamdetection import preprocess
import util

data_str = "spam_2024" # ["spam_626", "spam_2024"]
model_str = "nb" #["nb", "lr"]
haveDetails = True
haveTag = True

if data_str == "spam_2024":
    dict_text2label = {}
    csvFile = open("rawdata/{}.csv".format(data_str), "r")
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
if data_str == "spam_626":
    dict_text2label = {}
    workbook=xlrd.open_workbook("rawdata/{}.xlsx".format(data_str))
    worksheet=workbook.sheet_by_index(0)
    nrows=worksheet.nrows
    for i in range(nrows):
        row_list = worksheet.row_values(i)
        text = row_list[0].replace('\n', ' ')
        text = text.strip()
        if text == "":
            continue
        if row_list[1] not in dict_text2label:
            dict_text2label[row_list[1]] = [text]
        else:
            dict_text2label[row_list[1]].append(text)

print("classes {}".format(len(dict_text2label)))


train_data = {}
test_data = {}
id_doc = 0
for k, v in dict_text2label.items():
    print("{} : {}".format(k, len(v)))
    train_x, test_x, train_y, test_y = train_test_split(v, [k for _ in range(len(v))], test_size=0.9, random_state=1)
    for t in train_x:
        train_data[id_doc] = {"text":util.clean_chinese_text(t), "tag":k}
        id_doc += 1
    for t in test_x:
        test_data[id_doc] = {"text":util.clean_chinese_text(t), "tag":k}
        id_doc += 1
print(len(train_data), len(test_data))

# training = preprocess.preprocess.cutWord(train_data, istraining=True)
# testing = preprocess.preprocess.cutWord(test_data, istraining=True)

# testing_items = testing.items()
testing_items = test_data.items()
data_test = [v["text"] for k, v in testing_items]
# training_items = training.items()
training_items = train_data.items()
data_train = [v["text"] for k, v in training_items]
train_label = [1 if v["tag"] == "垃圾内容" else 0 for k, v in training_items]

tf_vectorizer = CountVectorizer(token_pattern="(?u)\\b\\w+\\b", binary=True)
# tf_vectorizer = CountVectorizer(lowercase=True, binary=True)
tf_vectorizer.fit(data_train+data_test)
train_features_tf = tf_vectorizer.transform(data_train) 
test_features_tf = tf_vectorizer.transform(data_test)

tfidf_vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
# tfidf_vectorizer = TfidfVectorizer(lowercase=True)
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

if model_str == "nb":
    model = MultinomialNB()
    model.fit(train_features_tfidf,train_label)
    pre = model.predict(test_features_tfidf)

    # model.fit(train_features_tf,train_label)
    # pre = model.predict(test_features_tf)
if model_str == "lr":
    model = LogisticRegression(C=7)
    model.fit(train_svd_scl,train_label)
    pre = model.predict(test_svd_scl)


res = {}
for idx, (id_,v) in enumerate(testing_items):
    d = {}
    d["content"] = v["text"]
    # if haveDetails:
    #     d["dict"] = v["dict"]
    #     d["list"] = v["list"]
    # if haveTag:
    d["tag"] = v["tag"]
    if pre[idx] == 1:
        d["suggest_tag"] = "垃圾内容"
    else:
        d["suggest_tag"] = "普通内容"
    res[id_] = d

pre_data = []
true_data = []
for id_, d in res.items():
    if d["suggest_tag"] == "垃圾内容":
        pre_data.append(1)
    else:
        pre_data.append(0)
    if d["tag"] == "垃圾内容":
        true_data.append(1)
    else:
        true_data.append(0)
    
prec, recall, f1, _ = precision_recall_fscore_support(true_data, pre_data, average="weighted")  
acc = accuracy_score(true_data, pre_data)
auc = roc_auc_score(true_data, pre_data)
print("prec:{} ; reacll:{} ; f1:{} ; acc:{} ; auc:{}".format(prec, recall, f1, acc, auc))


