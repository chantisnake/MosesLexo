# -*- coding: utf-8 -*-
from __future__ import division

# this program detects word anomaly using z-test for proportion
# assume the possibility of a particular word appear in a text follows normal distribution
from math import sqrt
from operator import itemgetter
from scipy.stats.stats import zprob
from extra import loadstastic


def ztest(p1, pt, n1, nt):
    """
    this method examine whether a particular word in a particular chunk is an anomaly compare to all rest of the chunks
    usually we think it is an anomaly if the return value is less than 0.05

    :param p1: the probability of a word's occurrence in a particular chunk:
                Number of word(the word we care about) occurrence in the chunk/ total word count in the chunk

    :param pt: the probability of a word's occurrence in all the chunks(or the whole passage)
                Number of word(the word we care about) occurrence in all the chunk/ total word count in all the chunk

    :param n1: the number word in the chunk we care about (total word count)
    :param nt: the number word in all the chunk selected (total word count)
    :return: the probability that the particular word in a particular chunk is NOT an anomaly
    """

    p = (p1 * n1 + pt * nt) / (n1 + nt)
    try:
        standard_error = sqrt(p * (1 - p) * ((1 / n1) + (1 / nt)))
        # print 'standard_error:', standard_error
        z_scores = (p1 - pt) / standard_error
        # print 'z_score', z_scores
        p_values = (1 - zprob(abs(z_scores))) * 2
        # print 'p_value:', p_values
        return p_values
    except:
        return 'Insignificant'


def merge_list(wordlists):
    """
    this function merges all the wordlist(dictionary) into one, and return it

    :param wordlists: an array contain all the wordlist(dictionary type)
    :return: the merged word list (dictionary type)
    """
    mergelist = {}
    for wordlist in wordlists:
        for key in wordlist.keys():
            try:
                mergelist[key] += wordlist[key]
            except:
                mergelist.update({key: wordlist[key]})
    return mergelist


def testall(WordLists, option='CustomP', Low=0.0, High=1.0):
    """
    this method takes Wordlist and and then analyze each single word(*compare to the total passage(all the chunks)*),
    and then pack that into the return

    :param WordLists:   Array
                        each element of array represent a chunk, and it is a dictionary type
                        each element in the dictionary maps word inside that chunk to its frequency

    :param option:  some default option to set for High And Low(see the document for High and Low)
                    1. using standard deviation to find outlier
                        TopStdE: only analyze the Right outlier of word, determined by standard deviation
                                    (word frequency > average + 2 * Standard_Deviation)
                        MidStdE: only analyze the Non-Outlier of word, determined by standard deviation
                                    (average + 2 * Standard_Deviation > word frequency > average - 2 * Standard_Deviation)
                        LowStdE: only analyze the Left Outlier of word, determined by standard deviation
                                    (average - 2 * Standard_Deviation > word frequency)

                    2. using IQR to find outlier *THIS METHOD DO NOT WORK WELL, BECAUSE THE DATA USUALLY ARE HIGHLY SKEWED*
                        TopIQR: only analyze the Top outlier of word, determined by IQR
                                    (word frequency > median + 1.5 * Standard)
                        MidIQR: only analyze the non-outlier of word, determined by IQR
                                    (median + 1.5 * Standard > word frequency > median - 1.5 * Standard)
                        LowIQR: only analyze the Left outlier of word, determined by IQR
                                    (median - 1.5 * Standard > word frequency)

    :param Low:  this method will only analyze the word with higher frequency than this value
                    (this parameter will be overwritten if the option is not 'Custom')
    :param High: this method will only analyze the word with lower frequency than this value
                    (this parameter will be overwritten if the option is not 'Custom')

    :return:    contain a array
                each element of array is a array, represent a chunk and it is sorted via p_value
                each element array is a tuple: (word, corresponding p_value)
    """

    # init
    MergeList = merge_list(WordLists)
    AllResults = []  # the value to return
    TotalWordCount = sum(MergeList.values())
    NumWord = len(MergeList)

    # option
    if option == 'CustomP':
        pass

    elif option == 'CustomF':
        Low /= NumWord
        High /= NumWord

    elif option.endswith('StdE'):
        StdE = 0
        Average = TotalWordCount / NumWord
        for word in MergeList:
            StdE += (MergeList[word] - Average) ** 2
        StdE = sqrt(StdE)
        StdE /= NumWord

        if option.startswith('Top'):
            # TopStdE: only analyze the Right outlier of word, determined by standard deviation
            Low = (Average + 2 * StdE) / NumWord

        elif option.startswith('Mid'):
            # MidStdE: only analyze the Non-Outlier of word, determined by standard deviation
            High = (Average + 2 * StdE) / NumWord
            Low = (Average - 2 * StdE) / NumWord

        elif option.startswith('Low'):
            # LowStdE: only analyze the Left Outlier of word, determined by standard deviation
            High = (Average - 2 * StdE) / NumWord

        else:
            print('input error')
            exit(-1)

    elif option.endswith('IQR'):
        TempList = sorted(MergeList.items(), key=itemgetter(1))
        Mid = TempList[int(NumWord / 2)][1]
        Q3 = TempList[int(NumWord * 3 / 4)][1]
        Q1 = TempList[int(NumWord / 4)][1]
        IQR = Q3 - Q1

        if option.startswith('Top'):
            # TopIQR: only analyze the Top outlier of word, determined by IQR
            Low = (Mid + 1.5 * IQR) / TotalWordCount

        elif option.startswith('Mid'):
            # MidIQR: only analyze the non-outlier of word, determined by IQR
            High = (Mid + 1.5 * IQR) / TotalWordCount
            Low = (Mid - 1.5 * IQR) / TotalWordCount

        elif option.startswith('Low'):
            # LowIQR: only analyze the Left outlier of word, determined by IQR
            High = (Mid - 1.5 * IQR) / TotalWordCount

        else:
            print('input error')
            exit(-1)

    else:
        print('input error')
        exit(-1)

    # calculation
    for wordlist in WordLists:
        ResultList = {}
        ListWordCount = sum(wordlist.values())

        for word in wordlist.keys():
            if Low < MergeList[word] / TotalWordCount < High:
                p_value = ztest(wordlist[word] / ListWordCount, MergeList[word] / TotalWordCount,
                                ListWordCount, TotalWordCount)
                ResultList.update({word: p_value})

        ResultList = sorted(ResultList.items(), key=itemgetter(1))
        AllResults.append(ResultList)

    return AllResults


def sort(word_p_lists):
    """
    this method combine all the diction in word_p_list(word with its p_value) into totallist,
    with a mark to indicate which file the element(word with p_value) belongs to
    and then sort the totallist, to give user a clean output of which word in which file is the most abnormal

    :param word_p_lists: a array of dictionary
                            each element of array represent a chunk, and it is a dictionary type
                            each element in the dictionary maps word inside that chunk to its p_value
    :return: a array of tuple type (sorted via p_value):
                each element is a tuple:    (the chunk it belong(the number of chunk in the word_p_list),
                                            the word, the corresponding p_value)

    """
    totallist = []
    i = 0
    for list in word_p_lists:
        templist = []
        for word in list:
            if not word[1] == 'Insignificant':
                temp = ('junk', i + 1) + word  # add the 'junk' to make i+1 a tuple type
                temp = temp[1:]
                templist.append(temp)
        totallist += templist
        i += 1

    totallist = sorted(totallist, key=lambda tup: tup[2])

    return totallist


def groupdivision(WordLists, ChunkMap):
    # Chunk test, make sure no two chunk are the same
    for i in range(len(ChunkMap)):
        for j in range(i + 1, len(ChunkMap)):
            if ChunkMap[i] == ChunkMap[j]:
                print 'Chunk ' + str(i) + ' and Chunk ' + str(j) + ' is the same'
                raise Exception

    # pack the Chunk data in to ChunkMap(because this is fast)
    for i in range(len(ChunkMap)):
        for j in range(len(ChunkMap[i])):
            ChunkMap[i][j] = WordLists[ChunkMap[i][j]]
    return ChunkMap


def testgroup(GroupWordLists, option='CustomP', Low=0.0, High=1.0):
    """
    this method takes ChunkWordlist and and then analyze each single word(compare to all the other group),
    and then pack that into the return

    :param GroupWordLists:   Array
                        each element of array represent a chunk, and it is a dictionary type
                        each element in the dictionary maps word inside that chunk to its frequency

    :param option:  some default option to set for High And Low(see the document for High and Low)
                    1. using standard deviation to find outlier
                        TopStdE: only analyze the Right outlier of word, determined by standard deviation
                                    (word frequency > average + 2 * Standard_Deviation)
                        MidStdE: only analyze the Non-Outlier of word, determined by standard deviation
                                    (average + 2 * Standard_Deviation > word frequency > average - 2 * Standard_Deviation)
                        LowStdE: only analyze the Left Outlier of word, determined by standard deviation
                                    (average - 2 * Standard_Deviation > word frequency)

                    2. using IQR to find outlier *THIS METHOD DO NOT WORK WELL, BECAUSE THE DATA USUALLY ARE HIGHLY SKEWED*
                        TopIQR: only analyze the Top outlier of word, determined by IQR
                                    (word frequency > median + 1.5 * Standard)
                        MidIQR: only analyze the non-outlier of word, determined by IQR
                                    (median + 1.5 * Standard > word frequency > median - 1.5 * Standard)
                        LowIQR: only analyze the Left outlier of word, determined by IQR
                                    (median - 1.5 * Standard > word frequency)

    :param Low:  this method will only analyze the word with higher frequency than this value
                    (this parameter will be overwritten if the option is not 'Custom')
    :param High: this method will only analyze the word with lower frequency than this value
                    (this parameter will be overwritten if the option is not 'Custom')

    :return:    contain a array
                each element of array is a array, represent a chunk and it is sorted via p_value
                each element array is a tuple: (word, corresponding p_value)
    """

    # init
    GroupLists = []
    GroupWordCounts = []
    GroupNumWords = []
    for Chunk in GroupWordLists:
        GroupLists.append(merge_list(Chunk))
        GroupWordCounts.append(sum(GroupLists[-1].values()))
        GroupNumWords.append(len(GroupLists[-1]))
    TotalList = merge_list(GroupLists)
    TotalWordCount = sum(GroupWordCounts)
    TotalNumWords = len(TotalList)
    AllResults = {}  # the value to return

    # option
    if option == 'CustomP':
        pass

    elif option == 'CustomF':
        Low /= TotalWordCount
        High /= TotalWordCount

    elif option.endswith('StdE'):
        StdE = 0
        Average = TotalWordCount / TotalNumWords
        for word in TotalList:
            StdE += (TotalList[word] - Average) ** 2
        StdE = sqrt(StdE)
        StdE /= TotalNumWords

        if option.startswith('Top'):
            # TopStdE: only analyze the Right outlier of word, determined by standard deviation
            Low = (Average + 2 * StdE) / TotalNumWords

        elif option.startswith('Mid'):
            # MidStdE: only analyze the Non-Outlier of word, determined by standard deviation
            High = (Average + 2 * StdE) / TotalNumWords
            Low = (Average - 2 * StdE) / TotalNumWords

        elif option.startswith('Low'):
            # LowStdE: only analyze the Left Outlier of word, determined by standard deviation
            High = (Average - 2 * StdE) / TotalNumWords

        else:
            print('input error')
            exit(-1)

    elif option.endswith('IQR'):
        TempList = sorted(TotalList.items(), key=itemgetter(1))
        Mid = TempList[int(TotalNumWords / 2)][1]
        Q3 = TempList[int(TotalNumWords * 3 / 4)][1]
        Q1 = TempList[int(TotalNumWords / 4)][1]
        IQR = Q3 - Q1

        if option.startswith('Top'):
            # TopIQR: only analyze the Top outlier of word, determined by IQR
            Low = (Mid + 1.5 * IQR) / TotalWordCount

        elif option.startswith('Mid'):
            # MidIQR: only analyze the non-outlier of word, determined by IQR
            High = (Mid + 1.5 * IQR) / TotalWordCount
            Low = (Mid - 1.5 * IQR) / TotalWordCount

        elif option.startswith('Low'):
            # LowIQR: only analyze the Left outlier of word, determined by IQR
            High = (Mid - 1.5 * IQR) / TotalWordCount

        else:
            print('input error')
            exit(-1)

    else:
        print('input error')
        exit(-1)

    # calculation
    for i in range(len(GroupWordLists)):
        for j in range(len(GroupWordLists)):
            if i != j:
                wordlistnumber = 0
                for wordlist in GroupWordLists[i]:
                    # print 'wordlists', wordlist
                    for word in wordlist.keys():
                        iWordCount = wordlist[word]
                        iTotalWordCount = sum(wordlist.values())
                        iWordProp = iWordCount / iTotalWordCount
                        try:
                            jWordCount = GroupLists[j][word]
                        except:
                            jWordCount = 0
                        jTotalWordCount = GroupWordCounts[j]
                        jWordProp = jWordCount / jTotalWordCount
                        if Low < iWordProp < High:
                            p_value = ztest(iWordProp, jWordProp, iTotalWordCount, jTotalWordCount)
                            # print iWordProp, jWordProp, iWordCount, jWordCount
                            try:
                                AllResults[(i, wordlistnumber, j)].append((word, p_value))
                            except:
                                AllResults.update({(i, wordlistnumber, j): [(word, p_value)]})
                    wordlistnumber += 1
    # sort the output
    for tuple in AllResults.keys():
        list = AllResults[tuple]
        list = sorted(list, key=lambda tup: tup[1])
        AllResults.update({tuple: list})
    return AllResults


if __name__ == "__main__":
    Wordlists = []

    for i in range(1, 12):
        f = open(str(i) + '.txt', 'r')
        content = f.read()
        f.close()
        Wordlist = loadstastic(content)
        Wordlists.append(Wordlist)

    Result = testall(Wordlists, option='TopStdE')

    for wordlist in Result:
        print(wordlist)
    print
    print "the list of most significants:"
    print sort(Result)

    print
    ChunkWordList = groupdivision(Wordlists, [[2, 3, 5], [4, 10, 6], [2, 4, 5]])
    '''
    for list in ChunkWordList:
        for dict in list:
            print 'dict', dict
            '''

    for item in testgroup(ChunkWordList).keys():
        print item, testgroup(ChunkWordList)[item]
