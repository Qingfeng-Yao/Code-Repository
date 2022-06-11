import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score

import dataclass

parser = argparse.ArgumentParser(description='using supervised machine learning')
parser.add_argument(
    '--dataset',
    default='Heyspam',
    help='Heyspam')
parser.add_argument(
    '--model', default='lr', help='nb | lr')
args = parser.parse_args()

dataset = getattr(dataclass, args.dataset)(is_deep=False, is_jieba=True, is_balanced=True)
 
tfidf_vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
tfidf_vectorizer.fit(dataset.text_corpus)
train_features_tfidf = tfidf_vectorizer.transform(dataset.text_train) 
test_features_tfidf = tfidf_vectorizer.transform(dataset.text_test)

if args.model == "nb":
    model = MultinomialNB()
elif args.model == "lr":
    model = LogisticRegression(C=7)

model.fit(train_features_tfidf,dataset.label_train)
pre = model.predict(test_features_tfidf)
# pre_pro = model.predict_proba(test_features_tfidf)
 
prec, recall, f1, _ = precision_recall_fscore_support(dataset.label_test, pre, average="weighted")  
acc = accuracy_score(dataset.label_test, pre)
auc = roc_auc_score(dataset.label_test, pre)
print("prec:{} ; reacll:{} ; f1:{} ; acc:{} ; auc:{}".format(prec, recall, f1, acc, auc))
