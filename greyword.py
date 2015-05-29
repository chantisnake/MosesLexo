# -*- coding:utf-8 -*-
from extra import loadstastic, creatdendro
from numpy.random import choice
from collections import defaultdict
import operator


def greyword(Contents, inittestrounds=100, significance=0.5):
    # load statics
    Len = len(Contents)
    WordLists = []
    ChunkSizes = []
    for DendroContent in Contents:
        wordlist = loadstastic(DendroContent)
        WordLists.append(wordlist)
        ChunkSizes.append(sum(wordlist.values()))

    # find largest file
    Max = 0
    for size in ChunkSizes:
        if Max < size:
            Max = size

    # create geryword and dendrogram for inittestround times
    DendroNum = {}
    FileStore = defaultdict()
    for _ in range(inittestrounds):
        # creates a deep copy of the lists
        tempWordLists = WordLists[:]
        tempChunkSizes = ChunkSizes[:]
        tempContents = Contents[:]

        for i in range(Len):  # focus on a file to normalize
            Diff = Max - ChunkSizes[i]
            NumGreyWord = Diff / Len

            for j in range(Len):  # choose a file to pick grey word
                wordlist = Contents[j].split()

                for _ in range(NumGreyWord):  # fill in the word
                    word = choice(wordlist)  # randomly choose a word from wordlist
                    tempContents[i] += ' ' + word
                    tempChunkSizes[i] += 1  # update chunksize
                    # update WordLists
                    try:
                        tempWordLists[i][word] += 1
                    except KeyError:
                        tempWordLists[i].update({word: 1})
        dendro = creatdendro(tempWordLists, tempChunkSizes)['leaves']
        dendro = tuple(dendro)  # convert the dendro into a tuple type
        # update DendroNum and FileStore
        try:
            DendroNum[dendro] += 1
            FileStore[dendro].append(tempContents)
        except KeyError:
            DendroNum.update({dendro: 1})
            FileStore.update({dendro: [tempContents]})

    # find the most common dendrogram
    print DendroNum
    ResultDendro = max(DendroNum.iteritems(), key=operator.itemgetter(1))[0]
    DendroContent = FileStore[ResultDendro]
    return ResultDendro, DendroContent


if __name__ == "__main__":
    Content = ["1 4 5 7 9 7 3", '1', '1', '1 4 5 7 9 7 3']
    tree, contents = greyword(Content, inittestrounds=100)
    print tree
    print contents
