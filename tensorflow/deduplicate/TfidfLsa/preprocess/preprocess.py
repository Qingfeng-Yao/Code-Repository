# -*- coding: utf-8 -*-
from .. import setting
from . import cut_word

def cutWord(raw_data):
    data = {}
    c = cut_word.CutWord(setting.STOPWORDS_PATH)
    for id_, text in raw_data.items():
        content = text["content"]
        d = {}
        d["title"] = text["title"]
        d["text"] = c.deal(text["title"] + " " + content)
        data[id_] = d
    return data