# -*- coding:utf-8 -*-
from extra import loadstastic, creatdendro
from numpy.random import choice
from collections import defaultdict
import operator
import os


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
        print 'the', _+1, 'times to test'
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
    ResultDendro = max(DendroNum.iteritems(), key=operator.itemgetter(1)) #[0]
    DendroContent = FileStore[ResultDendro[0]]
    return ResultDendro, DendroContent


if __name__ == "__main__":
    count = 0
    for _ in range(100):
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
        ResultDendro, DendroContent = greyword(Content, inittestrounds=1000)
        print 'calculation done'
        if ResultDendro[1] != 1:
            print 'different'
            count += 1
    print(count)


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
