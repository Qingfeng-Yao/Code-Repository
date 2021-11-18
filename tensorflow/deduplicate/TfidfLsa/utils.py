# -*- coding: utf-8 -*-
import os

from . import setting


def readStopwords(path=os.path.join(setting.STATIC_PATH, "stopwords.txt")):
    stopwords = []
    with open(path, "r", encoding="utf-8") as f:
        stopwords = [word.strip() for word in f.readlines()]
    return stopwords