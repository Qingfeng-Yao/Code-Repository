# -*- coding: utf-8 -*-
from os import path


################
# Project Configuration
################

# Path to project
BASE_PATH = path.dirname(path.realpath(__file__))

# Path to static files such as stopwords
STATIC_PATH = path.join(BASE_PATH, "static")

STOPWORDS_PATH = path.join(STATIC_PATH, "stopwords.txt")
