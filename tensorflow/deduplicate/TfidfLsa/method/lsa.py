# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
from collections import defaultdict

def tfidfTransform(texts):
    token_pattern = "(?u)\\b\\w+\\b"
    tfidf_vectorizer = TfidfVectorizer(token_pattern=token_pattern)
    tfidf_feature = tfidf_vectorizer.fit_transform(texts)
    return tfidf_feature, tfidf_vectorizer

def lsaTransform(X, n_components=1000):
    n_components = min(n_components, X.shape[-1])
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    lsa_feature = lsa.fit_transform(X)
    return lsa_feature, svd

def calcSimilarity(q_x, a_x):
    return q_x @ a_x.T



def calc(query, lsa_n_components, threshold):
    query_ids = []
    query_texts = []
    query_items = query.items()
    for idx, (id_, v) in enumerate(query_items):
        query_ids.append(id_)
        query_texts.append(v["text"])

    def getFeature(query_texts):
        texts = query_texts
        
        tfidf_feature, _ = tfidfTransform(texts)
        
        lsa_feature, _ = lsaTransform(tfidf_feature, n_components=lsa_n_components)

        query_feature = np.array(lsa_feature)
        return query_feature

    query_feature = getFeature(query_texts)

    similarity = calcSimilarity(query_feature, query_feature)

    bag = defaultdict(list)
    for query_idx, (q_id, v) in enumerate(query_items):
        if len(bag) == 0:
            bag[q_id].append(query_idx)
            continue
        
        score = 0
        sim_bid = ""
        for bid, group in bag.items():
            for idx in group:
                sim = similarity[query_idx, idx]
                if sim>score:
                    score = sim
                    sim_bid = bid
        if score>threshold:
            bag[sim_bid].append(query_idx)
        else:
            bag[q_id].append(query_idx)

    return bag, list(query_items)
