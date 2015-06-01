# -*- coding:utf-8 -*-
from __future__ import division
from extra import loadstastic, creatdendro
from copy import deepcopy
from numpy.random import choice
from collections import defaultdict
import operator
import os


def greyword(Contents):
    # load statics
    Len = len(Contents)
    WordLists = []
    ChunkSizes = []
    for DendroContent in Contents:
        wordlist = loadstastic(DendroContent)
        WordLists.append(wordlist)
        ChunkSizes.append(sum(wordlist.values()))
    Result = deepcopy(WordLists)

    # find largest file
    Max = 0
    for size in ChunkSizes:
        if Max < size:
            Max = size

    # create geryword and dendrogram
    for i in range(Len):  # focus on a file to normalize
        Diff = Max - ChunkSizes[i]
        NumGreyWord = Diff / (Len-1)  # not use itself

        for j in range(Len):  # choose a file to pick grey word
            if j != i and NumGreyWord != 0:  # not it self, or do not need to add
                for word in WordLists[j].keys():
                    # add word to the chunk
                    try:
                        Result[i][word] += WordLists[j][word]*(NumGreyWord/ChunkSizes[j])
                    except KeyError:
                        Result[i].update({word: WordLists[j][word]*(NumGreyWord/ChunkSizes[j])})

    dendro = creatdendro(WordLists, ChunkSizes)['leaves']
    return Result, dendro


if __name__ == "__main__":

    print 'hello'
    # read from 'TestSuite' folder
    path = os.path.join(os.getcwd(), 'TestSuite')
    data = {}
    for dir_entry in os.listdir(path):
        dir_entry_path = os.path.join(path, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r') as my_file:
                data[dir_entry] = my_file.read()
    Content = data.values()

    print 'content read'
    newWordLists, bingmeiyouniaoyong = greyword(Content)
    print
    for wordlist in newWordLists:
        for key in wordlist.keys():
            wordlist[key] = round(wordlist[key])
    for wordlist in newWordLists:
        print wordlist
        print sum(wordlist.values())




    '''
    # write to folders
    contentnum = 0
    for content in DendroContent:
        os.mkdir('example'+str(contentnum))
        save_path = os.path.join(os.getcwd(), 'example'+str(contentnum))
        for i in range(len(Content)):
            name_of_file = data.keys()[i]
            completeName = os.path.join(save_path, name_of_file)
            file1 = open(completeName, "w")
            file1.write(content[i])
            file1.close()
        contentnum += 1
        '''
