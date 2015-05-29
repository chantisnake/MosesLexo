# -*- coding:utf-8 -*-
from extra import loadstastic
def greyword(Contents, inittestrounds = 100, significance = 0.5):
    WordLists = []
    for content in Contents:
        wordlist = loadstastic(content)
        WordLists.append(wordlist)

