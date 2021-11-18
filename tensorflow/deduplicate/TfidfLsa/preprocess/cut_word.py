# -*- coding: utf-8 -*-
from .. import utils

import jieba


class CutWord(object):
    def __init__(self, stopwords_path):
        self.stopwords_path = stopwords_path
        self.stopwords = utils.readStopwords(stopwords_path)

    def addDict(self, dict_list):
        map(lambda x: jieba.load_userdict(x), dict_list)

    def deal(self, content):
        seg_list = jieba.cut(content.strip())
        ret = []
        for word in seg_list:
            word = word.strip()
            if word not in self.stopwords:
                if word != "\t" and word != "\n":
                    ret.append(word)
        return ' '.join(ret)
