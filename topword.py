# -*- coding: utf-8 -*-
from __future__ import division

# this program detects word anomaly using z-test for proportion
# assume the possibility of a particular word appear in a text follows normal distribution
# this program can be optimized in many way.
from math import sqrt
from operator import itemgetter
from scipy.stats.stats import zprob
from extra import loadstastic, merge_list, matrixtodict


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
                each element of array is a dictionary map a tuple to a list
                    tuple consist of 3 element (group number 1, list number, group number 2)
                        means compare the words in list number of group number 1 to all the word in group number 2
                    the list contain tuples, sorted by p value:
                        tuple means (word, p value)
                        this is word usage of word in group (group number 1), list (list number),
                        compare to the word usage of the same word in group (group number 2)
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
    a = [['', 'a', 'aban', 'abead', 'abrahame', 'abrahames', 'abrecan', 'ac', 'aceorfe\xc3\xb0', 'acol', 'acul',
          'acw\xc3\xa6\xc3\xb0', 'adzarias', 'afeallan', 'aferan', 'agan', 'agangen', 'aglac', 'ag\xc3\xa6f', 'ahicgan',
          'ahte', 'ahton', 'ahwearf', 'aldor', 'aldordom', 'aldordomes', 'aldorlege', 'aldre', 'alet', 'alhstede',
          'alwihta', 'alysde', 'al\xc3\xa6t', 'al\xc3\xa6ten', 'an', 'ana', 'and', 'andan', 'andsaca', 'andswaredon',
          'andswarode', 'ane', 'angin', 'anhydig', 'anmedlan', 'anmod', 'annanias', 'anne', 'anra', 'anwalh', 'ar',
          'are', 'areccan', 'arehte', 'arna', 'ar\xc3\xa6dan', 'ar\xc3\xa6dde', 'ar\xc3\xa6rde', 'asecganne', 'asette',
          'astah', 'astige\xc3\xb0', 'as\xc3\xa6gde', 'ateah', 'awacodon', 'awoc', 'aworpe', 'awunnen', 'azarias',
          'a\xc3\xbeencean', 'babilon', 'babilone', 'babilonia', 'babilonie', 'babilonige', 'baldazar', 'balde',
          'banum', 'baswe', 'be', 'beacen', 'beacne', 'bead', 'beam', 'beames', 'bearn', 'bearnum', 'bearwe', 'bebead',
          'bebodes', 'bebodo', 'bebuga\xc3\xb0', 'becwom', 'befolen', 'begete', 'belegde', 'belocene', 'bende', 'beon',
          'beorgas', 'beorht', 'beorhte', 'beorn', 'beornas', 'beot', 'beote', 'beran', 'bera\xc3\xb0', 'bere',
          'berhtmhwate', 'beseah', 'besn\xc3\xa6dan', 'besn\xc3\xa6ded', 'beswac', 'besw\xc3\xa6led', 'beteran',
          'bewindan', 'bewr\xc3\xa6con', 'be\xc3\xbeeahte', 'bidda\xc3\xb0', 'billa', 'bitera', 'bi\xc3\xb0', 'blacan',
          'blace', 'bleda', 'bledum', 'bletsian', 'bletsia\xc3\xb0', 'bletsie', 'bletsige', 'blican', 'bli\xc3\xb0e',
          'bli\xc3\xb0emod', 'bli\xc3\xb0emode', 'bl\xc3\xa6d', 'bl\xc3\xa6de', 'bl\xc3\xa6dum', 'boca', 'bocerum',
          'bocstafas', 'bolgenmod', 'bote', 'bradne', 'brandas', 'brego', 'brema\xc3\xb0', 'breme',
          'breostge\xc3\xb0ancum', 'breostlocan', 'bresne', 'brimfaro\xc3\xbees', 'brohte', 'brungen', 'bryne',
          'brytnedon', 'bryttedon', 'br\xc3\xa6con', 'br\xc3\xa6sna', 'bude', 'bun', 'burga', 'burge', 'burh', 'burhge',
          'burhsittende', 'burhsittendum', 'burhware', 'burnon', 'butan', 'bylywit', 'byman', 'byrig', 'byrnende',
          'b\xc3\xa6d', 'b\xc3\xa6don', 'b\xc3\xa6lblyse', 'b\xc3\xa6le', 'b\xc3\xa6r', 'b\xc3\xa6rnan', 'b\xc3\xa6ron',
          'caldea', 'caldeas', 'can', 'ceald', 'ceapian', 'ceastergeweorc', 'ceastre', 'cempan', 'cenned', 'ceorfan',
          'clammum', 'cl\xc3\xa6ne', 'cneomagum', 'cneorissum', 'cneow', 'cneowum', 'cnihta', 'cnihtas', 'cnihton',
          'cnihtum', 'com', 'come', 'comon', 'cor\xc3\xb0res', 'cr\xc3\xa6ft', 'cr\xc3\xa6ftas', 'cuman', 'cumble',
          'cunnode', 'cunnon', 'curon', 'cu\xc3\xb0', 'cu\xc3\xb0on', 'cu\xc3\xb0ost', 'cwale', 'cwealme', 'cwelm',
          'cwe\xc3\xb0an', 'cwe\xc3\xb0a\xc3\xb0', 'cwom', 'cwome', 'cw\xc3\xa6don', 'cw\xc3\xa6\xc3\xb0', 'cyme',
          'cymst', 'cyn', 'cynegode', 'cyne\xc3\xb0rymme', 'cynig', 'cyning', 'cyningdom', 'cyningdome', 'cyninge',
          'cyninges', 'cyrdon', 'cyst', 'cy\xc3\xb0an', 'daga', 'daniel', 'deaw', 'dea\xc3\xb0', 'dea\xc3\xb0e', 'dema',
          'deoflu', 'deoflum', 'deofolwitgan', 'deopne', 'deor', 'deora', 'deormode', 'deorum', 'derede', 'de\xc3\xb0',
          'diran', 'dom', 'domas', 'dome', 'domige', 'don', 'dreag', 'dreamas', 'dreame', 'dreamleas', 'drearung',
          'drihten', 'drihtenweard', 'drihtne', 'drihtnes', 'drincan', 'dropena', 'drugon', 'dryge', 'duge\xc3\xb0e',
          'duge\xc3\xbeum', 'dugu\xc3\xb0e', 'dyde', 'dydon', 'dyglan', 'dygle', 'd\xc3\xa6da', 'd\xc3\xa6de',
          'd\xc3\xa6dhwatan', 'd\xc3\xa6g', 'd\xc3\xa6ge', 'd\xc3\xa6ges', 'eac', 'eacenne', 'eacne', 'ead', 'eadmodum',
          'eagum', 'ealdfeondum', 'ealdor', 'ealdormen', 'ealhstede', 'eall', 'ealle', 'ealles', 'eallum', 'ealne',
          'ealra', 'earce', 'eard', 'eare', 'earfo\xc3\xb0m\xc3\xa6cg', 'earfo\xc3\xb0si\xc3\xb0as', 'earme', 'earmra',
          'earmre', 'earmsceapen', 'eart', 'eastream', 'ea\xc3\xb0medum', 'ebrea', 'ece', 'ecgum', 'ecne', 'edsceafte',
          'efnde', 'efndon', 'efne', 'eft', 'egesa', 'egesan', 'egesful', 'egeslic', 'egeslicu', 'egle', 'ehtode',
          'ende', 'ended\xc3\xa6g', 'endelean', 'engel', 'englas', 'engles', 'eode', 'eodon', 'eorla', 'eorlas',
          'eorlum', 'eor\xc3\xb0an', 'eor\xc3\xb0buendum', 'eor\xc3\xb0cyninga', 'eor\xc3\xb0lic', 'eowed', 'eower',
          'esnas', 'est', 'e\xc3\xb0el', 'facne', 'fandedon', 'fea', 'feax', 'fela', 'feld', 'felda', 'feohsceattum',
          'feonda', 'feondas', 'feore', 'feorh', 'feorhnere', 'feorum', 'feor\xc3\xb0a', 'feower', 'feran', 'findan',
          'fleam', 'fleon', 'folc', 'folca', 'folce', 'folcgesi\xc3\xb0um', 'folcm\xc3\xa6gen', 'folctoga', 'folctogan',
          'folcum', 'foldan', 'for', 'foran', 'forbr\xc3\xa6con', 'forburnene', 'foremihtig', 'forfangen', 'forgeaf',
          'forht', 'forh\xc3\xa6fed', 'forlet', 'forstas', 'for\xc3\xb0am', 'for\xc3\xb0on', 'for\xc3\xbeam',
          'fraco\xc3\xb0', 'fram', 'frasade', 'frea', 'freagleawe', 'frean', 'frecnan', 'frecne', 'fremde', 'fremede',
          'freobearn', 'freo\xc3\xb0o', 'fri\xc3\xb0', 'fri\xc3\xb0e', 'fri\xc3\xb0es', 'frod', 'frofre', 'frumcyn',
          'frumgaras', 'frumsl\xc3\xa6pe', 'frumspr\xc3\xa6ce', 'fr\xc3\xa6gn', 'fr\xc3\xa6twe', 'fuglas', 'fugolas',
          'funde', 'fundon', 'fur\xc3\xb0or', 'fyl', 'fyll', 'fyr', 'fyre', 'fyrenan', 'fyrend\xc3\xa6dum', 'fyrene',
          'fyrenum', 'fyres', 'fyrndagum', 'fyrstmearc', 'f\xc3\xa6c', 'f\xc3\xa6der', 'f\xc3\xa6gre', 'f\xc3\xa6r',
          'f\xc3\xa6rgryre', 'f\xc3\xa6st', 'f\xc3\xa6stan', 'f\xc3\xa6ste', 'f\xc3\xa6stlicne', 'f\xc3\xa6stna',
          'f\xc3\xa6stne', 'f\xc3\xa6\xc3\xb0m', 'f\xc3\xa6\xc3\xb0me', 'f\xc3\xa6\xc3\xb0mum', 'gad', 'gang', 'gangan',
          'gange', 'gast', 'gasta', 'gastas', 'gaste', 'gastes', 'gastum', 'ge', 'gealhmod', 'gealp', 'gearo', 'gearu',
          'gebead', 'gebearh', 'gebede', 'gebedu', 'gebedum', 'gebindan', 'gebletsad', 'gebletsige', 'geboden',
          'geborgen', 'geb\xc3\xa6don', 'gecoren', 'gecorene', 'gecw\xc3\xa6don', 'gecw\xc3\xa6\xc3\xb0',
          'gecy\xc3\xb0', 'gecy\xc3\xb0de', 'gecy\xc3\xb0ed', 'gedemed', 'gedon', 'gedydon', 'geegled', 'geflymed',
          'gefrecnod', 'gefremede', 'gefrigen', 'gefrunon', 'gefr\xc3\xa6ge', 'gefr\xc3\xa6gn', 'gef\xc3\xa6gon',
          'gegleded', 'gegnunga', 'gehete', 'gehogode', 'gehwam', 'gehwearf', 'gehwilc', 'gehwilcum', 'gehwurfe',
          'gehw\xc3\xa6s', 'gehydum', 'gehyge', 'gehyrdon', 'geleafan', 'gelic', 'gelicost', 'gelimpan', 'gelyfan',
          'gelyfde', 'gelyfest', 'gel\xc3\xa6dde', 'gel\xc3\xa6ste', 'gemenged', 'gemet', 'gemunan', 'gemunde',
          'gemynd', 'gemyndgast', 'gem\xc3\xa6ne', 'gem\xc3\xa6ted', 'gem\xc3\xa6tte', 'genamon', 'generede', 'gengum',
          'genumen', 'geoca', 'geoce', 'geocre', 'geocrostne', 'geogo\xc3\xb0e', 'geond', 'geondsawen', 'geonge',
          'georn', 'georne', 'gerefan', 'gerume', 'gerusalem', 'gerynu', 'gerysna', 'ger\xc3\xa6dum', 'gesawe',
          'gesawon', 'gesceaft', 'gesceafta', 'gesceafte', 'gesceod', 'gesceode', 'gescylde', 'geseah', 'geseald',
          'gesecganne', 'geseo', 'geseted', 'gese\xc3\xb0ed', 'gesigef\xc3\xa6ste', 'gesi\xc3\xb0', 'gesloh',
          'gespr\xc3\xa6c', 'gestreon', 'geswi\xc3\xb0de', 'gesyh\xc3\xb0e', 'ges\xc3\xa6de', 'ges\xc3\xa6ledne',
          'ges\xc3\xa6t', 'getenge', 'geteod', 'geteode', 'gewand', 'gewat', 'geweald', 'gewealde', 'gewear\xc3\xb0',
          'gewemman', 'gewemmed', 'geweox', 'gewindagum', 'gewit', 'gewita', 'gewittes', 'geworden', 'gewordene',
          'geworhte', 'gewur\xc3\xb0ad', 'gewur\xc3\xb0od', 'gewyrhto', 'ge\xc3\xb0afian', 'ge\xc3\xb0anc',
          'ge\xc3\xb0ances', 'ge\xc3\xb0ancum', 'ge\xc3\xb0enc', 'ge\xc3\xb0inges', 'ge\xc3\xbeanc', 'ge\xc3\xbeeahte',
          'ge\xc3\xbeingu', 'gif', 'gife', 'gifena', 'ginge', 'gingum', 'glade', 'gleaw', 'gleawmode', 'gleawost',
          'gleda', 'gl\xc3\xa6dmode', 'god', 'gode', 'godes', 'godspellode', 'gods\xc3\xa6de', 'gold', 'golde',
          'goldfatu', 'gramlice', 'grene', 'grim', 'grimman', 'grimme', 'grimmost', 'grome', 'grund', 'grynde\xc3\xb0',
          'gryre', 'gr\xc3\xa6s', 'gulpon', 'guman', 'gumena', 'gumrices', 'gumum', 'gyddedon', 'gyddigan', 'gyfe',
          'gyfum', 'gyld', 'gyldan', 'gylde', 'gyldnan', 'gylp', 'gylpe', 'g\xc3\xa6delingum', 'g\xc3\xa6st', 'habban',
          'habba\xc3\xb0', 'had', 'hade', 'hale', 'halegu', 'halga', 'halgan', 'halgum', 'halig', 'halige', 'haliges',
          'haligra', 'haligu', 'hamsittende', 'hand', 'hat', 'hata', 'hatan', 'haten', 'hatne', 'hatte', 'he', 'hea',
          'heah', 'heahbyrig', 'heahcyning', 'heahheort', 'healdan', 'healda\xc3\xb0', 'healle', 'hean', 'heane',
          'heanne', 'heapum', 'hearan', 'hearde', 'hearm', 'hebbanne', 'hefonfugolas', 'hegan', 'heh', 'help', 'helpe',
          'helpend', 'heofenum', 'heofnum', 'heofona', 'heofonas', 'heofonbeorht', 'heofones', 'heofonheane',
          'heofonrices', 'heofonsteorran', 'heofontunglum', 'heofonum', 'heold', 'heolde', 'heora', 'heorta', 'heortan',
          'heorugrimra', 'here', 'herede', 'heredon', 'herega', 'heretyma', 'herewosan', 'hergas', 'herga\xc3\xb0',
          'hergende', 'heriga\xc3\xb0', 'herige', 'heriges', 'herran', 'het', 'hete', 'heton', 'hie', 'hige',
          'higecr\xc3\xa6ft', 'hige\xc3\xbeancle', 'him', 'hine', 'his', 'hit', 'hlaford', 'hleo', 'hleo\xc3\xb0or',
          'hleo\xc3\xb0orcwyde', 'hleo\xc3\xb0orcyme', 'hleo\xc3\xb0rade', 'hlifigan', 'hlifode', 'hliga\xc3\xb0',
          'hluttor', 'hlypum', 'hlyst', 'hofe', 'hogedon', 'hold', 'holt', 'hordm\xc3\xa6gen', 'horsce', 'hra\xc3\xb0e',
          'hra\xc3\xb0or', 'hreddan', 'hremde', 'hreohmod', 'hre\xc3\xb0', 'hrof', 'hrofe', 'hrusan', 'hryre',
          'hr\xc3\xa6gle', 'hu', 'huslfatu', 'hwa', 'hwalas', 'hwearf', 'hweorfan', 'hwilc', 'hwile', 'hwurfan',
          'hwurfon', 'hwyrft', 'hw\xc3\xa6t', 'hw\xc3\xa6\xc3\xb0ere', 'hw\xc3\xa6\xc3\xb0re', 'hyge', 'hyld',
          'hyldelease', 'hyldo', 'hyllas', 'hyra', 'hyran', 'hyrde', 'hyrdon', 'hyrra', 'hyrran', 'hyssas',
          'h\xc3\xa6fde', 'h\xc3\xa6fdest', 'h\xc3\xa6fdon', 'h\xc3\xa6ft', 'h\xc3\xa6ftas', 'h\xc3\xa6le\xc3\xb0',
          'h\xc3\xa6le\xc3\xb0a', 'h\xc3\xa6le\xc3\xb0um', 'h\xc3\xa6to', 'h\xc3\xa6\xc3\xb0en', 'h\xc3\xa6\xc3\xb0ena',
          'h\xc3\xa6\xc3\xb0enan', 'h\xc3\xa6\xc3\xb0endom', 'h\xc3\xa6\xc3\xb0ene', 'h\xc3\xa6\xc3\xb0enra',
          'h\xc3\xa6\xc3\xb0ne', 'h\xc3\xa6\xc3\xb0num', 'iacobe', 'ic', 'ican', 'in', 'inge\xc3\xbeancum', 'innan',
          'inne', 'is', 'isaace', 'isen', 'iserne', 'isernum', 'israela', 'iudea', 'lacende', 'lafe', 'lagon',
          'lagostreamas', 'landa', 'landgesceaft', 'lange', 'lare', 'larum', 'la\xc3\xb0', 'la\xc3\xb0e',
          'la\xc3\xb0searo', 'lean', 'leas', 'leng', 'lengde', 'leoda', 'leode', 'leodum', 'leofum', 'leoge\xc3\xb0',
          'leoht', 'leohtfruma', 'leohtran', 'leoman', 'leornedon', 'let', 'lice', 'lif', 'lifde', 'life', 'lifes',
          'liffrean', 'liffruman', 'lifgende', 'lifigea\xc3\xb0', 'lifigen', 'lifigende', 'lig', 'lige', 'liges',
          'ligetu', 'ligeword', 'ligges', 'lignest', 'lisse', 'li\xc3\xb0', 'locia\xc3\xb0', 'locode', 'lof',
          'lofia\xc3\xb0', 'lofige', 'lufan', 'lufia\xc3\xb0', 'lust', 'lyfte', 'lyftlacende', 'lyhte', 'lytel',
          'l\xc3\xa6g', 'ma', 'magon', 'man', 'mancynne', 'mandreame', 'mandrihten', 'mandrihtne', 'mane', 'manegum',
          'manig', 'manlican', 'manna', 'mannum', 'mara', 'mare', 'me', 'meaht', 'meahte', 'meda', 'medugal', 'medum',
          'meld', 'men', 'menigo', 'merestreamas', 'meted', 'metod', 'metode', 'metodes', 'me\xc3\xb0elstede',
          'me\xc3\xb0le', 'micel', 'micelne', 'miclan', 'micle', 'mid', 'middangeard', 'middangeardes', 'migtigra',
          'miht', 'mihta', 'mihte', 'mihtig', 'mihtigran', 'mihton', 'mihtum', 'miltse', 'miltsum', 'min', 'mine',
          'minra', 'minsode', 'mirce', 'misael', 'mod', 'mode', 'modge\xc3\xb0anc', 'modge\xc3\xbeances', 'modhwatan',
          'modig', 'modsefan', 'modum', 'moldan', 'mona', 'monig', 'monige', 'mores', 'mor\xc3\xb0re', 'moste',
          'myndga\xc3\xb0', 'm\xc3\xa6', 'm\xc3\xa6cgum', 'm\xc3\xa6ge', 'm\xc3\xa6gen', 'm\xc3\xa6genes',
          'm\xc3\xa6lmete', 'm\xc3\xa6nige', 'm\xc3\xa6nigeo', 'm\xc3\xa6re', 'm\xc3\xa6rost', 'm\xc3\xa6st',
          'm\xc3\xa6tinge', 'm\xc3\xa6tra', 'na', 'nabochodonossor', 'nacod', 'nales', 'nalles', 'nama', 'naman',
          'name', 'ne', 'neata', 'neh', 'nehstum', 'neod', 'nerede', 'nergend', 'nergenne', 'nerigende', 'niht', 'nis',
          'ni\xc3\xb0', 'ni\xc3\xb0a', 'ni\xc3\xb0as', 'ni\xc3\xb0hete', 'ni\xc3\xb0wracum', 'no', 'noldon', 'nu',
          'nydde', 'nyde', 'nydgenga', 'nym\xc3\xb0e', 'nym\xc3\xbee', 'ny\xc3\xb0or', 'n\xc3\xa6nig', 'n\xc3\xa6ron',
          'n\xc3\xa6s', 'of', 'ofen', 'ofer', 'oferfaren', 'oferf\xc3\xa6\xc3\xb0mde', 'oferhogedon', 'oferhyd',
          'oferhygd', 'oferhygde', 'oferhygdum', 'ofermedlan', 'ofestum', 'ofn', 'ofne', 'ofnes', 'ofstlice', 'oft',
          'oftor', 'on', 'oncw\xc3\xa6\xc3\xb0', 'onegdon', 'onfenge', 'onfon', 'ongan', 'ongeald', 'ongeat', 'onget',
          'onginnan', 'ongunnon', 'ongyt', 'onhicga\xc3\xb0', 'onhnigon', 'onhwearf', 'onhweorfe\xc3\xb0',
          'onh\xc3\xa6tan', 'onh\xc3\xa6ted', 'onlah', 'onm\xc3\xa6lde', 'onsended', 'onsoce', 'onsocon', 'onsteallan',
          'ontreowde', 'onwoc', 'or', 'ord', 'ordfruma', 'orlegra', 'orl\xc3\xa6g', 'owiht', 'owihtes', 'o\xc3\xb0',
          'o\xc3\xb0er', 'o\xc3\xb0stod', 'o\xc3\xb0\xc3\xb0e', 'o\xc3\xb0\xc3\xbe\xc3\xa6t', 'persum', 'reccan',
          'reccend', 'regna', 'rehte', 'reordberend', 'reorde', 'rest', 'reste', 'restende', 're\xc3\xb0e', 'rica',
          'rice', 'rices', 'riht', 'rihte', 'rihtne', 'rodera', 'roderum', 'rodora', 'rodorbeorhtan', 'rodore',
          'rohton', 'rume', 'run', 'runcr\xc3\xa6ftige', 'ryne', 'r\xc3\xa6d', 'r\xc3\xa6dan', 'r\xc3\xa6das',
          'r\xc3\xa6df\xc3\xa6st', 'r\xc3\xa6dleas', 'r\xc3\xa6rde', 'r\xc3\xa6swa', 'salomanes', 'samnode', 'samod',
          'sand', 'sawla', 'sawle', 'sceal', 'scealcas', 'sceatas', 'sceod', 'sceolde', 'sceoldon', 'scima',
          'scine\xc3\xb0', 'scufan', 'scur', 'scyde', 'scylde', 'scyldig', 'scyppend', 'scyrede', 'se', 'sealde',
          'sealte', 'sealtne', 'secan', 'secgan', 'secge', 'sefa', 'sefan', 'sel', 'seld', 'sele', 'sellende', 'sende',
          'sended', 'sende\xc3\xb0', 'sendon', 'sennera', 'seo', 'seofan', 'seofon', 'seolfes', 'seon', 'septon',
          'settend', 'sidne', 'sie', 'sien', 'siendon', 'sigora', 'sin', 'sine', 'sines', 'sinne', 'sinra', 'sinum',
          'si\xc3\xb0', 'si\xc3\xb0estan', 'si\xc3\xb0f\xc3\xa6t', 'si\xc3\xb0ian', 'si\xc3\xb0\xc3\xb0an', 'sloh',
          'sl\xc3\xa6pe', 'snawas', 'snotor', 'snytro', 'snyttro', 'sohton', 'somod', 'sona', 'sorge', 'sorh',
          'so\xc3\xb0', 'so\xc3\xb0an', 'so\xc3\xb0cwidum', 'so\xc3\xb0e', 'so\xc3\xb0f\xc3\xa6st',
          'so\xc3\xb0f\xc3\xa6stra', 'so\xc3\xb0ra', 'so\xc3\xb0um', 'sped', 'spel', 'spelboda', 'spelbodan',
          'spowende', 'spreca\xc3\xb0', 'spr\xc3\xa6c', 'starude', 'sta\xc3\xb0ole', 'stefn', 'stefne', 'stigan',
          'stille', 'stod', 'stode', 'strudon', 'sum', 'sumera', 'sumeres', 'sumor', 'sundor', 'sundorgife', 'sungon',
          'sunna', 'sunnan', 'sunne', 'sunu', 'susl', 'swa', 'swefen', 'swefn', 'swefnede', 'swefnes', 'sweg',
          'swelta\xc3\xb0', 'swigode', 'swilce', 'swi\xc3\xb0', 'swi\xc3\xb0an', 'swi\xc3\xb0e', 'swi\xc3\xb0mod',
          'swi\xc3\xb0rian', 'swi\xc3\xb0rode', 'swutol', 'swylc', 'swylce', 'sw\xc3\xa6f', 'syle', 'sylf', 'sylfa',
          'sylfe', 'sylle', 'symble', 'syndon', 's\xc3\xa6de', 's\xc3\xa6don', 's\xc3\xa6faro\xc3\xb0a', 's\xc3\xa6gde',
          's\xc3\xa6gdon', 's\xc3\xa6t', 's\xc3\xa6ton', 's\xc3\xa6w\xc3\xa6gas', 'tacen', 'tacna', 'telgum', 'tempel',
          'teode', 'teodest', 'teonfullum', 'teso', 'tid', 'tida', 'tide', 'tirum', 'to', 'todrifen',
          'todw\xc3\xa6sced', 'tohworfene', 'torhtan', 'tosceaf', 'tosomne', 'toswende', 'tosweop', 'towrecene',
          'treddedon', 'treow', 'treowum', 'trymede', 'tunglu', 'twigum', 'ufan', 'unbli\xc3\xb0e', 'unceapunga',
          'under', 'ungelic', 'ungescead', 'unlytel', 'unriht', 'unrihtdom', 'unrihtum', 'unrim', 'unr\xc3\xa6d',
          'unscyndne', 'unwaclice', 'up', 'upcyme', 'uppe', 'us', 'user', 'usic', 'ut', 'utan', 'wage', 'waldend',
          'wall', 'wandode', 'wast', 'wa\xc3\xb0e', 'we', 'wealde\xc3\xb0', 'wealla', 'wealle', 'weard', 'weardas',
          'weardode', 'wearmlic', 'wear\xc3\xb0', 'wece\xc3\xb0', 'wecga\xc3\xb0', 'weder', 'wedera', 'wedere', 'weg',
          'welan', 'wendan', 'wende', 'weoh', 'weold', 'weorca', 'weor\xc3\xb0e\xc3\xb0', 'wer', 'wera', 'weras',
          'wereda', 'werede', 'weredes', 'werigra', 'weroda', 'werode', 'werodes', 'werum', 'wer\xc3\xb0eode', 'wes',
          'wesan', 'westen', 'wiccungdom', 'widan', 'wide', 'wideferh\xc3\xb0', 'widne', 'widost', 'wig', 'wiges',
          'wihgyld', 'wiht', 'wihte', 'wildan', 'wilddeor', 'wilddeora', 'wilddeorum', 'wildeora', 'wildra', 'wildu',
          'willa', 'willan', 'willa\xc3\xb0', 'wille', 'wilnedan', 'wilnian', 'winburge', 'winde', 'windig',
          'windruncen', 'wine', 'wineleasne', 'wingal', 'winter', 'winterbiter', 'wintra', 'wis', 'wisa', 'wisdom',
          'wise', 'wislice', 'wisne', 'wisse', 'wiste', 'wiston', 'wite', 'witegena', 'witga', 'witgode', 'witgum',
          'witig', 'witiga\xc3\xb0', 'witigdom', 'witod', 'wi\xc3\xb0', 'wi\xc3\xb0erbreca', 'wlancan', 'wlenco',
          'wlite', 'wlitescyne', 'wlitig', 'wlitiga', 'wod', 'wodan', 'wolcenfaru', 'wolcna', 'wolde', 'wolden',
          'woldon', 'wom', 'woma', 'woman', 'womma', 'worce', 'word', 'worda', 'wordcwide', 'wordcwyde', 'worde',
          'worden', 'wordgleaw', 'wordum', 'worhton', 'world', 'worlde', 'worn', 'woruld', 'woruldcr\xc3\xa6fta',
          'worulde', 'woruldgesceafta', 'woruldlife', 'woruldrice', 'woruldspedum', 'wrace', 'wrat', 'wrece\xc3\xb0',
          'writan', 'write', 'wroht', 'wr\xc3\xa6c', 'wr\xc3\xa6cca', 'wr\xc3\xa6clic', 'wr\xc3\xa6stran', 'wudu',
          'wudubeam', 'wudubeames', 'wuldor', 'wuldorcyning', 'wuldorf\xc3\xa6st', 'wuldorhaman', 'wuldre', 'wuldres',
          'wulfheort', 'wunast', 'wunden', 'wundor', 'wundorlic', 'wundra', 'wundre', 'wundrum', 'wunian',
          'wunia\xc3\xb0', 'wunode', 'wurde', 'wurdon', 'wurpon', 'wur\xc3\xb0an', 'wur\xc3\xb0edon',
          'wur\xc3\xb0ia\xc3\xb0', 'wur\xc3\xb0igean', 'wur\xc3\xb0myndum', 'wylla', 'wylm', 'wynsum', 'wyrcan', 'wyrd',
          'wyrda', 'wyrrestan', 'wyrtruma', 'wyrtruman', 'wyrtum', 'w\xc3\xa6da', 'w\xc3\xa6de', 'w\xc3\xa6fran',
          'w\xc3\xa6g', 'w\xc3\xa6re', 'w\xc3\xa6rf\xc3\xa6ste', 'w\xc3\xa6rgenga', 'w\xc3\xa6ron', 'w\xc3\xa6s',
          'w\xc3\xa6ter', 'w\xc3\xa6terscipe', 'w\xc3\xa6tersprync', 'yfel', 'ylda', 'yldran', 'yldum', 'ymb', 'ymbe',
          'yrre', 'ywed', 'y\xc3\xb0a', '\xc3\xa6', '\xc3\xa6cr\xc3\xa6ftig', '\xc3\xa6fre', '\xc3\xa6fter',
          '\xc3\xa6f\xc3\xa6ste', '\xc3\xa6ghw\xc3\xa6s', '\xc3\xa6ht', '\xc3\xa6hta', '\xc3\xa6hte', '\xc3\xa6lbeorht',
          '\xc3\xa6led', '\xc3\xa6lmihtig', '\xc3\xa6lmihtiges', '\xc3\xa6lmihtigne', '\xc3\xa6lmyssan', '\xc3\xa6nig',
          '\xc3\xa6r', '\xc3\xa6rendbec', '\xc3\xa6renum', '\xc3\xa6rest', '\xc3\xa6t', '\xc3\xa6tb\xc3\xa6r',
          '\xc3\xa6te', '\xc3\xa6tywed', '\xc3\xa6\xc3\xb0ele', '\xc3\xa6\xc3\xb0eling', '\xc3\xa6\xc3\xb0elinga',
          '\xc3\xa6\xc3\xb0elingas', '\xc3\xa6\xc3\xb0elinge', '\xc3\xa6\xc3\xb0elum', '\xc3\xb0a', '\xc3\xb0am',
          '\xc3\xb0ara', '\xc3\xb0e', '\xc3\xb0eah', '\xc3\xb0eaw', '\xc3\xb0ec', '\xc3\xb0eoda', '\xc3\xb0eode',
          '\xc3\xb0eoden', '\xc3\xb0eodne', '\xc3\xb0eonydum', '\xc3\xb0on', '\xc3\xb0one', '\xc3\xb0onne',
          '\xc3\xb0ry', '\xc3\xb0u', '\xc3\xb0uhte', '\xc3\xb0urhgleded', '\xc3\xb0y', '\xc3\xb0\xc3\xa6r',
          '\xc3\xb0\xc3\xa6re', '\xc3\xb0\xc3\xa6s', '\xc3\xbea', '\xc3\xbeafigan', '\xc3\xbeam', '\xc3\xbean',
          '\xc3\xbeanc', '\xc3\xbeancia\xc3\xb0', '\xc3\xbeancode', '\xc3\xbeara', '\xc3\xbeas', '\xc3\xbee',
          '\xc3\xbeeah', '\xc3\xbeeaw', '\xc3\xbeec', '\xc3\xbeegn', '\xc3\xbeegnas', '\xc3\xbeegnum', '\xc3\xbeeh',
          '\xc3\xbeenden', '\xc3\xbeeode', '\xc3\xbeeoden', '\xc3\xbeeodne', '\xc3\xbeeodnes', '\xc3\xbeeostro',
          '\xc3\xbeeowned', '\xc3\xbeider', '\xc3\xbein', '\xc3\xbeine', '\xc3\xbeinga', '\xc3\xbeinne', '\xc3\xbeis',
          '\xc3\xbeisse', '\xc3\xbeolia\xc3\xb0', '\xc3\xbeone', '\xc3\xbeonne', '\xc3\xberage', '\xc3\xberah',
          '\xc3\xbereanied', '\xc3\xbereaum', '\xc3\xberidda', '\xc3\xberidde', '\xc3\xberistum', '\xc3\xberowigean',
          '\xc3\xberowode', '\xc3\xbery', '\xc3\xberym', '\xc3\xberymmes', '\xc3\xbeu', '\xc3\xbeuhte', '\xc3\xbeurfe',
          '\xc3\xbeurh', '\xc3\xbeurhwodon', '\xc3\xbey', '\xc3\xbeyder', '\xc3\xbeysne', '\xc3\xbe\xc3\xa6r',
          '\xc3\xbe\xc3\xa6re', '\xc3\xbe\xc3\xa6s', '\xc3\xbe\xc3\xa6t', '\xc3\xbe\xc3\xa6tte'],
         ['A1.3_Dan_T00030_1', 0.00074229691876750699, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00051075476061984433, 0.0081844327945245383,
          0.00024743230625583565, 0.0021008403361344537, 0.00025537738030992216, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0042016806722689074,
          0.00025537738030992216, 0.00050280968656575781, 0.00025537738030992216, 0.0022745612006270402,
          0.00025537738030992216, 0.00025537738030992216, 0.0021008403361344537, 0.00074229691876750699,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.0025957049486461251, 0.0020032936722089449,
          0.042034667077840315, 0.00050280968656575781, 0.00025537738030992216, 0.0021008403361344537,
          0.0026036500227002118, 0.00049486461251167129, 0.0021008403361344537, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.0025957049486461251, 0.00049486461251167129,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0023482726423902896, 0.00025537738030992216, 0.00024743230625583565, 0.00025537738030992216,
          0.00025537738030992216, 0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565,
          0.0023482726423902896, 0.00024743230625583565, 0.0021008403361344537, 0.00025537738030992216,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00025537738030992216,
          0.0025957049486461251, 0.0021008403361344537, 0.00025537738030992216, 0.0095588663599996578,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00051075476061984433,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.0023482726423902896,
          0.00024743230625583565, 0.00050280968656575781, 0.00025537738030992216, 0.00098972922502334258,
          0.00024743230625583565, 0.00074229691876750699, 0.0023562177164443763, 0.00024743230625583565,
          0.0023482726423902896, 0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565,
          0.0009976742990774291, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00049486461251167129, 0.00025537738030992216, 0.0021008403361344537, 0.00049486461251167129,
          0.00024743230625583565, 0.00024743230625583565, 0.00051075476061984433, 0.0023482726423902896,
          0.00051075476061984433, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00074229691876750699, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0023482726423902896, 0.00025537738030992216, 0.00024743230625583565, 0.0015084290596972735,
          0.00024743230625583565, 0.00049486461251167129, 0.0021008403361344537, 0.00024743230625583565,
          0.00051075476061984433, 0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.0021008403361344537,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.0021008403361344537, 0.00098972922502334258, 0.00025537738030992216,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.0030985146352118832,
          0.00025537738030992216, 0.00076613214092976644, 0.00024743230625583565, 0.00075818706687568003,
          0.00025537738030992216, 0.00024743230625583565, 0.00076613214092976644, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129,
          0.0028510823289560473, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.0025957049486461251, 0.0031064597092659691, 0.00075818706687568003, 0.00025537738030992216,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00025537738030992216,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00025537738030992216, 0.00024743230625583565, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.0046965452847805801, 0.00024743230625583565,
          0.00049486461251167129, 0.0026115950967542981, 0.00024743230625583565, 0.0023562177164443763,
          0.0021008403361344537, 0.0030985146352118832, 0.00074229691876750699, 0.00050280968656575781,
          0.00024743230625583565, 0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565,
          0.00049486461251167129, 0.0023482726423902896, 0.00025537738030992216, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.0028510823289560473, 0.00050280968656575781, 0.00024743230625583565, 0.0012530516793873513,
          0.00049486461251167129, 0.00024743230625583565, 0.00025537738030992216, 0.00049486461251167129,
          0.00025537738030992216, 0.00024743230625583565, 0.010654087323002596, 0.00025537738030992216,
          0.00024743230625583565, 0.0044491129785247437, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565, 0.0064524066507336882,
          0.00049486461251167129, 0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565,
          0.00025537738030992216, 0.00025537738030992216, 0.0021008403361344537, 0.00024743230625583565,
          0.00098972922502334258, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0057021646579120955,
          0.00075818706687568003, 0.0025957049486461251, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.0023482726423902896, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.0084318651007803721, 0.00024743230625583565, 0.0030985146352118832,
          0.0010056193731315156, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216, 0.0023482726423902896,
          0.00098972922502334258, 0.00024743230625583565, 0.00024743230625583565, 0.0021008403361344537,
          0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565, 0.0023482726423902896,
          0.00050280968656575781, 0.00024743230625583565, 0.001484593837535014, 0.00024743230625583565,
          0.0021008403361344537, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216,
          0.0028431372549019606, 0.0035933792477235542, 0.00049486461251167129, 0.00098972922502334258,
          0.00049486461251167129, 0.00025537738030992216, 0.00025537738030992216, 0.00051075476061984433,
          0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216, 0.0021008403361344537,
          0.00024743230625583565, 0.0021008403361344537, 0.00025537738030992216, 0.0014925389115891005,
          0.00024743230625583565, 0.00024743230625583565, 0.0025957049486461251, 0.0019953485981548582,
          0.00025537738030992216, 0.00074229691876750699, 0.0021008403361344537, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.0017638064400071955, 0.00024743230625583565,
          0.003353892015521805, 0.0021008403361344537, 0.00025537738030992216, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.0044491129785247437, 0.00025537738030992216,
          0.00024743230625583565, 0.0043356761664910606, 0.00024743230625583565, 0.00051075476061984433,
          0.0021008403361344537, 0.00024743230625583565, 0.00051075476061984433, 0.00049486461251167129,
          0.00075024199282159351, 0.0031064597092659691, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00051075476061984433, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.0012371615312791783, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216, 0.00024743230625583565,
          0.0021008403361344537, 0.00049486461251167129, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00050280968656575781, 0.0023562177164443763,
          0.00025537738030992216, 0.00024743230625583565, 0.00075818706687568003, 0.0009976742990774291,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00051075476061984433,
          0.0023482726423902896, 0.00025537738030992216, 0.00074229691876750699, 0.0075999665921414441,
          0.0025957049486461251, 0.00025537738030992216, 0.00024743230625583565, 0.00025537738030992216,
          0.00025537738030992216, 0.00050280968656575781, 0.00025537738030992216, 0.0021008403361344537,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00049486461251167129, 0.00024743230625583565, 0.00074229691876750699, 0.00025537738030992216,
          0.00098972922502334258, 0.0021008403361344537, 0.0026036500227002118, 0.00024743230625583565,
          0.00074229691876750699, 0.00024743230625583565, 0.0021008403361344537, 0.00049486461251167129,
          0.00024743230625583565, 0.00024743230625583565, 0.00050280968656575781, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.0021008403361344537,
          0.0021008403361344537, 0.00024743230625583565, 0.0023482726423902896, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0021008403361344537,
          0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565, 0.00098972922502334258,
          0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.0024743230625583566, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.0021008403361344537, 0.00025537738030992216,
          0.0026036500227002118, 0.00024743230625583565, 0.0015084290596972735, 0.00074229691876750699,
          0.00049486461251167129, 0.00050280968656575781, 0.0026036500227002118, 0.00024743230625583565,
          0.010759579060982192, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.0023562177164443763, 0.0021008403361344537, 0.0023482726423902896, 0.0010056193731315156,
          0.00024743230625583565, 0.00050280968656575781, 0.0028510823289560473, 0.00050280968656575781,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00050280968656575781, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00049486461251167129,
          0.0023482726423902896, 0.0012371615312791783, 0.00025537738030992216, 0.00024743230625583565,
          0.0025957049486461251, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00049486461251167129, 0.0021008403361344537,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129, 0.0021008403361344537,
          0.0023562177164443763, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.0021008403361344537, 0.00051075476061984433, 0.00098972922502334258,
          0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.0021008403361344537, 0.003338001867413632,
          0.00024743230625583565, 0.00049486461251167129, 0.0023482726423902896, 0.0009976742990774291,
          0.0021008403361344537, 0.00024743230625583565, 0.00025537738030992216, 0.0026115950967542981,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.0023482726423902896, 0.00025537738030992216, 0.0023482726423902896, 0.00075818706687568003,
          0.00050280968656575781, 0.00049486461251167129, 0.0017399712178449362, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00076613214092976644, 0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129,
          0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216, 0.0021008403361344537,
          0.00049486461251167129, 0.00024743230625583565, 0.0010056193731315156, 0.00025537738030992216,
          0.00050280968656575781, 0.0023482726423902896, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216, 0.00025537738030992216,
          0.00050280968656575781, 0.00025537738030992216, 0.00024743230625583565, 0.00050280968656575781,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.0023482726423902896, 0.0021008403361344537, 0.0021008403361344537,
          0.00049486461251167129, 0.00024743230625583565, 0.00050280968656575781, 0.00024743230625583565,
          0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565, 0.0055807827718243261,
          0.0051993549713463365, 0.0063786952089704384, 0.00025537738030992216, 0.0021008403361344537,
          0.00075818706687568003, 0.00049486461251167129, 0.00025537738030992216, 0.00025537738030992216,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00050280968656575781, 0.00049486461251167129, 0.00024743230625583565,
          0.00098972922502334258, 0.00024743230625583565, 0.00025537738030992216, 0.00098972922502334258,
          0.0020271288943712043, 0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0010135644471856021,
          0.00051075476061984433, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00049486461251167129, 0.00074229691876750699, 0.00024743230625583565,
          0.00051075476061984433, 0.00024743230625583565, 0.00098972922502334258, 0.0012371615312791783,
          0.0038328664799253034, 0.00049486461251167129, 0.0023562177164443763, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.0015322642818595329, 0.00024743230625583565,
          0.00024743230625583565, 0.00074229691876750699, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.025103778514463892, 0.00075818706687568003, 0.0009976742990774291,
          0.00025537738030992216, 0.00050280968656575781, 0.00024743230625583565, 0.00050280968656575781,
          0.00024743230625583565, 0.00051075476061984433, 0.0009976742990774291, 0.0021008403361344537,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0025957049486461251,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.0021008403361344537, 0.00075024199282159351, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00049486461251167129, 0.00024743230625583565, 0.00075024199282159351, 0.00025537738030992216,
          0.00024743230625583565, 0.0012371615312791783, 0.00024743230625583565, 0.0012451066053332648,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00074229691876750699,
          0.00025537738030992216, 0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.00098972922502334258, 0.0012451066053332648,
          0.00049486461251167129, 0.00024743230625583565, 0.0090322214512716416, 0.00025537738030992216,
          0.00024743230625583565, 0.015984161248597299, 0.0028510823289560473, 0.0021008403361344537,
          0.0021008403361344537, 0.042621017826090679, 0.0030905695611577965, 0.010937968459555077,
          0.0028431372549019606, 0.00025537738030992216, 0.00050280968656575781, 0.00024743230625583565,
          0.0023482726423902896, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00025537738030992216, 0.00025537738030992216, 0.00024743230625583565,
          0.00025537738030992216, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.0067973856209150325, 0.00051075476061984433,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.00050280968656575781, 0.0021008403361344537, 0.00049486461251167129,
          0.00024743230625583565, 0.0038567017020875627, 0.00074229691876750699, 0.00024743230625583565,
          0.00049486461251167129, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.0022348358303566074, 0.0023482726423902896, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.0017320261437908497,
          0.00075024199282159351, 0.00024743230625583565, 0.00075024199282159351, 0.00024743230625583565,
          0.00049486461251167129, 0.00075818706687568003, 0.0010056193731315156, 0.00024743230625583565,
          0.00024743230625583565, 0.00049486461251167129, 0.0023482726423902896, 0.0028431372549019606,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129,
          0.00024743230625583565, 0.00024743230625583565, 0.0057180548060202681, 0.00024743230625583565,
          0.022571634158250458, 0.00024743230625583565, 0.0009976742990774291, 0.00024743230625583565,
          0.0019794584500466856, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.003864646776141649, 0.00025537738030992216, 0.00024743230625583565,
          0.0044491129785247437, 0.00024743230625583565, 0.00024743230625583565, 0.00074229691876750699,
          0.00024743230625583565, 0.00050280968656575781, 0.00050280968656575781, 0.00024743230625583565,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00050280968656575781, 0.00024743230625583565, 0.00025537738030992216, 0.0030985146352118832,
          0.00075024199282159351, 0.00075818706687568003, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00049486461251167129,
          0.0021008403361344537, 0.00051075476061984433, 0.00049486461251167129, 0.00024743230625583565,
          0.0021008403361344537, 0.00025537738030992216, 0.00049486461251167129, 0.00024743230625583565,
          0.00025537738030992216, 0.00075024199282159351, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.001484593837535014, 0.00074229691876750699, 0.00049486461251167129,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.00025537738030992216, 0.00025537738030992216, 0.00024743230625583565,
          0.0021008403361344537, 0.00050280968656575781, 0.00051075476061984433, 0.00024743230625583565,
          0.0021008403361344537, 0.00025537738030992216, 0.00024743230625583565, 0.00074229691876750699,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00075818706687568003,
          0.00024743230625583565, 0.00024743230625583565, 0.0070448179271708688, 0.0021008403361344537,
          0.0023482726423902896, 0.00025537738030992216, 0.00025537738030992216, 0.00025537738030992216,
          0.00025537738030992216, 0.0041041340083433986, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.0027455905909764518, 0.0023482726423902896, 0.0019953485981548582,
          0.0021008403361344537, 0.00024743230625583565, 0.0012451066053332648, 0.00024743230625583565,
          0.00024743230625583565, 0.00050280968656575781, 0.0022507259784647804, 0.0026036500227002118,
          0.00024743230625583565, 0.00024743230625583565, 0.00075024199282159351, 0.00098972922502334258,
          0.0038408115539793905, 0.00074229691876750699, 0.00025537738030992216, 0.00051075476061984433,
          0.0009976742990774291, 0.00024743230625583565, 0.00024743230625583565, 0.0028510823289560473,
          0.0044491129785247437, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0025957049486461251, 0.0028510823289560473, 0.0010056193731315156, 0.00025537738030992216,
          0.0021008403361344537, 0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565,
          0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0021008403361344537,
          0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00025537738030992216, 0.00024743230625583565, 0.00049486461251167129,
          0.0047044903588346655, 0.0030985146352118832, 0.00025537738030992216, 0.00025537738030992216,
          0.0021008403361344537, 0.00025537738030992216, 0.00025537738030992216, 0.0010056193731315156,
          0.00025537738030992216, 0.0023482726423902896, 0.00024743230625583565, 0.00024743230625583565,
          0.00049486461251167129, 0.00024743230625583565, 0.024788117081695066, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00074229691876750699, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00049486461251167129, 0.0010135644471856021, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.0057101097319661818, 0.00074229691876750699,
          0.0038408115539793901, 0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216,
          0.00049486461251167129, 0.0023482726423902896, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.0049439775910364156, 0.0060835924583900842, 0.00049486461251167129,
          0.0079528906363768755, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00075818706687568003, 0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00074229691876750699, 0.001484593837535014,
          0.00024743230625583565, 0.00025537738030992216, 0.00074229691876750699, 0.00025537738030992216,
          0.033366591284831978, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.00049486461251167129, 0.0012451066053332648, 0.00024743230625583565, 0.0023482726423902896,
          0.00075024199282159351, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.0023482726423902896, 0.0021008403361344537, 0.0021008403361344537,
          0.0021008403361344537, 0.00025537738030992216, 0.00025537738030992216, 0.00074229691876750699,
          0.00024743230625583565, 0.0023482726423902896, 0.0021008403361344537, 0.00024743230625583565,
          0.0067973856209150333, 0.0056284532161488448, 0.00025537738030992216, 0.0023482726423902896,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0021008403361344537,
          0.00049486461251167129, 0.00024743230625583565, 0.0044570580525788292, 0.00024743230625583565,
          0.0023482726423902896, 0.00024743230625583565, 0.004375401536761493, 0.0028590274030101336,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.001500483985643187, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00050280968656575781, 0.00025537738030992216,
          0.00024743230625583565, 0.00049486461251167129, 0.00025537738030992216, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.0010056193731315156,
          0.00025537738030992216, 0.00024743230625583565, 0.00051075476061984433, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00049486461251167129,
          0.00024743230625583565, 0.00024743230625583565, 0.0073001953074807915, 0.0026115950967542981,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00074229691876750699,
          0.00024743230625583565, 0.00049486461251167129, 0.00049486461251167129, 0.00074229691876750699,
          0.0021008403361344537, 0.021430948526199471, 0.0030985146352118832, 0.00024743230625583565,
          0.00024743230625583565, 0.0025957049486461251, 0.0044491129785247437, 0.00025537738030992216,
          0.0026036500227002118, 0.007547627613736627, 0.00024743230625583565, 0.0028669724770642203,
          0.00025537738030992216, 0.00024743230625583565, 0.00074229691876750699, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00050280968656575781, 0.0046148887689632431,
          0.00024743230625583565, 0.0010056193731315156, 0.00024743230625583565, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129, 0.0009976742990774291,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.00025537738030992216,
          0.0025957049486461251, 0.0021008403361344537, 0.00024743230625583565, 0.0021008403361344537,
          0.0068053306949691196, 0.00025537738030992216, 0.00025537738030992216, 0.00025537738030992216,
          0.00025537738030992216, 0.005373075835838923, 0.00024743230625583565, 0.0025957049486461251,
          0.00024743230625583565, 0.0023562177164443763, 0.0025957049486461251, 0.00024743230625583565,
          0.00025537738030992216, 0.00049486461251167129, 0.0021008403361344537, 0.00024743230625583565,
          0.0021008403361344537, 0.003338001867413632, 0.0021008403361344537, 0.00024743230625583565,
          0.00049486461251167129, 0.0028510823289560473, 0.00024743230625583565, 0.00049486461251167129,
          0.00075024199282159351, 0.00049486461251167129, 0.00024743230625583565, 0.00050280968656575781,
          0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00049486461251167129, 0.00049486461251167129, 0.00049486461251167129,
          0.00024743230625583565, 0.00049486461251167129, 0.00075024199282159351, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00075818706687568003, 0.018485596073291703, 0.005191409897292251, 0.0026036500227002118,
          0.0021008403361344537, 0.0088982259570494875, 0.00024743230625583565, 0.0021008403361344537,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00075818706687568003, 0.005191409897292251, 0.00025537738030992216, 0.00025537738030992216,
          0.00024743230625583565, 0.00049486461251167129, 0.00049486461251167129, 0.0021008403361344537,
          0.00024743230625583565, 0.00025537738030992216, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00049486461251167129,
          0.0023482726423902896, 0.00024743230625583565, 0.0026115950967542981, 0.0023482726423902896,
          0.0021008403361344537, 0.00050280968656575781, 0.00024743230625583565, 0.00075024199282159351,
          0.00024743230625583565, 0.00074229691876750699, 0.00025537738030992216, 0.00024743230625583565,
          0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.022220915889291495,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.0025957049486461251, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129, 0.00049486461251167129,
          0.0042016806722689074, 0.00025537738030992216, 0.0012530516793873513, 0.0021008403361344537,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216,
          0.00025537738030992216, 0.0009976742990774291, 0.00024743230625583565, 0.00024743230625583565,
          0.00049486461251167129, 0.00074229691876750699, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.0019874035241007719, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00051075476061984433, 0.0043277310924369752,
          0.00050280968656575781, 0.00025537738030992216, 0.00025537738030992216, 0.010532705436914827,
          0.00025537738030992216, 0.00025537738030992216, 0.00024743230625583565, 0.011659706696134113,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00076613214092976644, 0.00025537738030992216,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.00049486461251167129, 0.00074229691876750699, 0.00076613214092976644, 0.0023482726423902896,
          0.0025957049486461251, 0.00050280968656575781, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565, 0.00051075476061984433,
          0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129, 0.00050280968656575781,
          0.0021008403361344537, 0.00025537738030992216, 0.0010215095212396887, 0.00024743230625583565,
          0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.0023482726423902896, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216, 0.00050280968656575781,
          0.00025537738030992216, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216,
          0.00025537738030992216, 0.00024743230625583565, 0.0021008403361344537, 0.00075818706687568003,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0012451066053332648,
          0.0063025210084033615, 0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565,
          0.0023482726423902896, 0.0023562177164443763, 0.00074229691876750699, 0.0036092693958317272,
          0.00025537738030992216, 0.0021008403361344537, 0.00024743230625583565, 0.0021008403361344537,
          0.00049486461251167129, 0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565,
          0.0019874035241007719, 0.00024743230625583565, 0.0021008403361344537, 0.00025537738030992216,
          0.00098972922502334258, 0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565,
          0.00025537738030992216, 0.00025537738030992216, 0.00024743230625583565, 0.00050280968656575781,
          0.0064603517247877754, 0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565,
          0.0042016806722689074, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0030905695611577965, 0.00075818706687568003, 0.00024743230625583565, 0.00050280968656575781,
          0.00024743230625583565, 0.0021008403361344537, 0.00024743230625583565, 0.0035933792477235542,
          0.00049486461251167129, 0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565,
          0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00025537738030992216, 0.00051075476061984433,
          0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.00024743230625583565,
          0.00074229691876750699, 0.00050280968656575781, 0.0044491129785247437, 0.00024743230625583565,
          0.00025537738030992216, 0.0027455905909764518, 0.00025537738030992216, 0.00024743230625583565,
          0.00024743230625583565, 0.0023482726423902896, 0.00049486461251167129, 0.00024743230625583565,
          0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0026036500227002118, 0.00049486461251167129, 0.00074229691876750699, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00074229691876750699, 0.00024743230625583565,
          0.00074229691876750699, 0.00050280968656575781, 0.0044491129785247437, 0.00049486461251167129,
          0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565, 0.00025537738030992216,
          0.0021008403361344537, 0.00024743230625583565, 0.00024743230625583565, 0.0085453019128140552,
          0.00024743230625583565, 0.00025537738030992216, 0.0029771327491241147, 0.023429573664328119,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.00024743230625583565, 0.0021008403361344537, 0.0012451066053332648,
          0.00024743230625583565, 0.00074229691876750699, 0.0021008403361344537, 0.00024743230625583565,
          0.0026036500227002118, 0.00050280968656575781, 0.00025537738030992216, 0.0036013243217776405,
          0.0025957049486461251, 0.0021008403361344537, 0.00050280968656575781, 0.00049486461251167129,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.001484593837535014,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216,
          0.0046148887689632431, 0.00025537738030992216, 0.00024743230625583565, 0.0023482726423902896,
          0.0010056193731315156, 0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.00024743230625583565, 0.00025537738030992216, 0.00025537738030992216,
          0.00075024199282159351, 0.00050280968656575781, 0.014571073077549061, 0.0024902132106665296,
          0.00025537738030992216, 0.0065943472190099287, 0.00075024199282159351, 0.00025537738030992216,
          0.0019794584500466852, 0.00050280968656575781, 0.00075024199282159351, 0.00074229691876750699,
          0.00024743230625583565, 0.00024743230625583565, 0.00049486461251167129, 0.00024743230625583565,
          0.00025537738030992216, 0.00024743230625583565, 0.00050280968656575781, 0.00024743230625583565,
          0.00024743230625583565, 0.0028590274030101336, 0.001484593837535014, 0.00075024199282159351,
          0.00075024199282159351, 0.056981921208850518, 0.00024743230625583565, 0.021807707792597164,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0021008403361344537,
          0.0028590274030101336, 0.00024743230625583565, 0.021576165634449501, 0.0035933792477235542,
          0.00024743230625583565, 0.0037114845938375348, 0.00024743230625583565, 0.00024743230625583565,
          0.0021008403361344537, 0.00024743230625583565, 0.0021008403361344537, 0.00074229691876750699,
          0.0023482726423902896, 0.0023482726423902896, 0.00049486461251167129, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.0012451066053332648, 0.00098972922502334258,
          0.00024743230625583565, 0.0023482726423902896, 0.00024743230625583565, 0.00024743230625583565,
          0.00024743230625583565, 0.0022348358303566078, 0.0032245650553799502, 0.00024743230625583565,
          0.00024743230625583565, 0.00024743230625583565, 0.00024743230625583565, 0.0021008403361344537,
          0.00025537738030992216, 0.00024743230625583565, 0.00024743230625583565, 0.00025537738030992216,
          0.0054388422035480874, 0.0021008403361344537, 0.00024743230625583565, 0.0073207539896692625,
          0.00074229691876750699, 0.00024743230625583565, 0.011035515123480586, 0.00024743230625583565,
          0.00049486461251167129, 0.00049486461251167129, 0.00024743230625583565, 0.013160190681777297,
          0.0052073000454004236, 0.0076657329598506076, 0.042320432760260068, 0.0073001953074807906],
         ['A1.3_Dan_T00030_2', 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0044444444444444444,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.017777777777777778, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0033333333333333335,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0044444444444444444, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0077777777777777776, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0,
          0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0055555555555555558, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0033333333333333335,
          0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0033333333333333335, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.012222222222222223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0033333333333333335, 0.0,
          0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0,
          0.0011111111111111111, 0.022222222222222223, 0.0011111111111111111, 0.0, 0.0, 0.017777777777777778,
          0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.012222222222222223, 0.0011111111111111111, 0.0033333333333333335,
          0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335,
          0.0033333333333333335, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0055555555555555558, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.016666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0033333333333333335,
          0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0033333333333333335, 0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0033333333333333335, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.022222222222222223,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.01, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.01,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0033333333333333335, 0.0, 0.0, 0.0, 0.0077777777777777776, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0077777777777777776,
          0.012222222222222223, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0,
          0.0022222222222222222, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0077777777777777776, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.017777777777777778,
          0.0011111111111111111, 0.016666666666666666, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.015555555555555555, 0.0044444444444444444, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0077777777777777776, 0.0022222222222222222, 0.0066666666666666671,
          0.018888888888888889, 0.0],
         ['A1.3_Dan_T00030_3', 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.06222222222222222,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0088888888888888889, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0,
          0.0055555555555555558, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0044444444444444444,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0022222222222222222, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0055555555555555558, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0044444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0033333333333333335, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0044444444444444444, 0.0022222222222222222, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.012222222222222223, 0.0, 0.0, 0.0, 0.01, 0.0011111111111111111,
          0.0055555555555555558, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.01, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111,
          0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0,
          0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0022222222222222222,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0022222222222222222,
          0.0033333333333333335, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0044444444444444444, 0.0011111111111111111,
          0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0055555555555555558, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.026666666666666668, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111,
          0.0022222222222222222, 0.0, 0.017777777777777778, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0044444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.012222222222222223, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0055555555555555558, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0, 0.0044444444444444444,
          0.0088888888888888889, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0088888888888888889,
          0.0044444444444444444, 0.0, 0.0077777777777777776, 0.0011111111111111111, 0.0, 0.0066666666666666671,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.017777777777777778, 0.0, 0.0088888888888888889,
          0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0066666666666666671,
          0.0, 0.0, 0.012222222222222223, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0022222222222222222, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0011111111111111111,
          0.01, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0055555555555555558, 0.0,
          0.0011111111111111111, 0.017777777777777778, 0.0022222222222222222],
         ['A1.3_Dan_T00030_4', 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0077777777777777776,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0033333333333333335, 0.026666666666666668, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0044444444444444444,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0044444444444444444, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0033333333333333335, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335,
          0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0022222222222222222, 0.0033333333333333335, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0066666666666666671, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.022222222222222223, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111,
          0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0055555555555555558, 0.0, 0.0, 0.0044444444444444444, 0.0011111111111111111,
          0.0, 0.0, 0.017777777777777778, 0.0022222222222222222, 0.012222222222222223, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0044444444444444444, 0.0, 0.0, 0.0, 0.0055555555555555558, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0044444444444444444, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0055555555555555558, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0055555555555555558, 0.0, 0.0033333333333333335, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.02, 0.0, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335, 0.0011111111111111111,
          0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0033333333333333335, 0.0,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.015555555555555555, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0011111111111111111,
          0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0022222222222222222,
          0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.012222222222222223, 0.0033333333333333335, 0.0011111111111111111, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0033333333333333335,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0077777777777777776, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0066666666666666671, 0.0, 0.0, 0.0,
          0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111,
          0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0055555555555555558,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0077777777777777776,
          0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0088888888888888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111,
          0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222,
          0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0,
          0.0022222222222222222, 0.0, 0.0055555555555555558, 0.0033333333333333335, 0.0, 0.0022222222222222222, 0.0,
          0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0,
          0.014444444444444444, 0.0, 0.0088888888888888889, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0088888888888888889, 0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111,
          0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0088888888888888889,
          0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0,
          0.0, 0.0077777777777777776, 0.0022222222222222222, 0.0, 0.0088888888888888889, 0.0011111111111111111,
          0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0077777777777777776, 0.026666666666666668,
          0.0011111111111111111],
         ['A1.3_Dan_T00030_5', 2.6758409785932719e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0022935779816513763, 0.0035900491695149009,
          8.9194699286442398e-06, 1.6864543982730707e-05, 0.0011467889908256881, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 3.3729087965461414e-05,
          0.0011467889908256881, 0.0011557084607543323, 0.0011467889908256881, 0.0069074923547400614,
          0.0011467889908256881, 0.0011467889908256881, 1.6864543982730707e-05, 2.6758409785932719e-05,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 3.4703483840019187e-05, 0.0034849643221202859,
          0.017029366792588596, 0.0011557084607543323, 0.0011467889908256881, 1.6864543982730707e-05,
          0.0011725730047370631, 1.783893985728848e-05, 1.6864543982730707e-05, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 3.4703483840019187e-05, 1.783893985728848e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          2.5784013911374943e-05, 0.0011467889908256881, 8.9194699286442398e-06, 0.0011467889908256881,
          0.0011467889908256881, 8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06,
          2.5784013911374943e-05, 8.9194699286442398e-06, 1.6864543982730707e-05, 0.0011467889908256881,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 0.0011467889908256881,
          3.4703483840019187e-05, 1.6864543982730707e-05, 0.0011467889908256881, 0.0058558943455057858,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0022935779816513763,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 2.5784013911374943e-05,
          8.9194699286442398e-06, 0.0011557084607543323, 0.0011467889908256881, 3.5677879714576959e-05,
          8.9194699286442398e-06, 2.6758409785932719e-05, 0.0011636535348084186, 8.9194699286442398e-06,
          2.5784013911374943e-05, 1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.001173547400611621, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.783893985728848e-05, 0.0011467889908256881, 1.6864543982730707e-05, 1.783893985728848e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0022935779816513763, 2.5784013911374943e-05,
          0.0022935779816513763, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 2.6758409785932719e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          2.5784013911374943e-05, 0.0011467889908256881, 8.9194699286442398e-06, 0.0034671253822629966,
          8.9194699286442398e-06, 1.783893985728848e-05, 1.6864543982730707e-05, 8.9194699286442398e-06,
          0.0022935779816513763, 8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 1.6864543982730707e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 1.6864543982730707e-05, 3.5677879714576959e-05, 0.0011467889908256881,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011904119445943515,
          0.0011467889908256881, 0.0034403669724770644, 8.9194699286442398e-06, 0.00230249745158002,
          0.0011467889908256881, 8.9194699286442398e-06, 0.0034403669724770644, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05,
          0.0011814924746657073, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          3.4703483840019187e-05, 0.0023282814654913952, 0.00230249745158002, 0.0011467889908256881,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 0.0011467889908256881,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 0.0011467889908256881, 8.9194699286442398e-06, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 5.1568027822749901e-05, 8.9194699286442398e-06,
          1.783893985728848e-05, 0.0023104425256341068, 8.9194699286442398e-06, 0.0011636535348084186,
          1.6864543982730707e-05, 0.0011904119445943515, 2.6758409785932719e-05, 0.0011557084607543323,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06,
          1.783893985728848e-05, 2.5784013911374943e-05, 0.0011467889908256881, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011814924746657075, 0.0011557084607543323, 8.9194699286442398e-06, 0.0023203363914373089,
          1.783893985728848e-05, 8.9194699286442398e-06, 0.0011467889908256881, 1.783893985728848e-05,
          0.0011467889908256881, 8.9194699286442398e-06, 0.0035613419679798525, 0.0011467889908256881,
          8.9194699286442398e-06, 4.2648557894105657e-05, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06, 0.0035276128800143911,
          1.783893985728848e-05, 8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06,
          0.0011467889908256881, 0.0011467889908256881, 1.6864543982730707e-05, 8.9194699286442398e-06,
          3.5677879714576959e-05, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0023629849493314145,
          0.00230249745158002, 3.4703483840019187e-05, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 2.5784013911374943e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0035989686394435451, 8.9194699286442398e-06, 0.0011904119445943519,
          0.0023114169215086642, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881, 2.5784013911374943e-05,
          3.5677879714576959e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.6864543982730707e-05,
          8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06, 2.5784013911374943e-05,
          0.0011557084607543323, 8.9194699286442398e-06, 5.3516819571865439e-05, 8.9194699286442398e-06,
          1.6864543982730707e-05, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881,
          4.362295376866343e-05, 0.0012082508844516401, 1.783893985728848e-05, 3.5677879714576959e-05,
          1.783893985728848e-05, 0.0011467889908256881, 0.0011467889908256881, 0.0022935779816513763,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881, 1.6864543982730707e-05,
          8.9194699286442398e-06, 1.6864543982730707e-05, 0.0011467889908256881, 0.0011913863404689094,
          8.9194699286442398e-06, 8.9194699286442398e-06, 3.4703483840019187e-05, 0.0023470948012232415,
          0.0011467889908256881, 2.6758409785932719e-05, 1.6864543982730707e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0046139143730886852, 8.9194699286442398e-06,
          0.0023372009354200398, 1.6864543982730707e-05, 0.0011467889908256881, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 4.2648557894105657e-05, 0.0011467889908256881,
          8.9194699286442398e-06, 0.0012350092942375727, 8.9194699286442398e-06, 0.0022935779816513763,
          1.6864543982730707e-05, 8.9194699286442398e-06, 0.0022935779816513763, 1.783893985728848e-05,
          0.0011646279306829768, 0.0023282814654913952, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0022935779816513763, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 4.4597349643221202e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881, 8.9194699286442398e-06,
          1.6864543982730707e-05, 1.783893985728848e-05, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011557084607543323, 0.0011636535348084186,
          0.0011467889908256881, 8.9194699286442398e-06, 0.00230249745158002, 0.001173547400611621,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0022935779816513763,
          2.5784013911374943e-05, 0.0011467889908256881, 2.6758409785932719e-05, 0.0081781795286922105,
          3.4703483840019187e-05, 0.0011467889908256881, 8.9194699286442398e-06, 0.0011467889908256881,
          0.0011467889908256881, 0.0011557084607543323, 0.0011467889908256881, 1.6864543982730707e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.783893985728848e-05, 8.9194699286442398e-06, 2.6758409785932719e-05, 0.0011467889908256881,
          3.5677879714576959e-05, 1.6864543982730707e-05, 0.0011725730047370631, 8.9194699286442398e-06,
          2.6758409785932719e-05, 8.9194699286442398e-06, 1.6864543982730707e-05, 1.783893985728848e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011557084607543323, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.6864543982730707e-05,
          1.6864543982730707e-05, 8.9194699286442398e-06, 2.5784013911374943e-05, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.6864543982730707e-05,
          1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 3.5677879714576959e-05,
          1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442405e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.6864543982730707e-05, 0.0011467889908256881,
          0.0011725730047370631, 8.9194699286442398e-06, 0.0034671253822629966, 2.6758409785932719e-05,
          1.783893985728848e-05, 0.0011557084607543323, 0.0011725730047370631, 8.9194699286442398e-06,
          0.0012311117107393415, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011636535348084186, 1.6864543982730707e-05, 2.5784013911374943e-05, 0.0023114169215086647,
          8.9194699286442398e-06, 0.0011557084607543323, 0.0011814924746657075, 0.0011557084607543323,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011557084607543323, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 1.783893985728848e-05,
          2.5784013911374943e-05, 4.4597349643221202e-05, 0.0011467889908256881, 8.9194699286442398e-06,
          3.4703483840019187e-05, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 1.783893985728848e-05, 1.6864543982730707e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05, 1.6864543982730707e-05,
          0.0011636535348084186, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 1.6864543982730707e-05, 0.0022935779816513763, 3.5677879714576959e-05,
          1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 1.6864543982730707e-05, 6.1461893625951909e-05,
          8.9194699286442398e-06, 1.783893985728848e-05, 2.5784013911374943e-05, 0.001173547400611621,
          1.6864543982730707e-05, 8.9194699286442398e-06, 0.0011467889908256881, 0.0023104425256341068,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          2.5784013911374943e-05, 0.0011467889908256881, 2.5784013911374943e-05, 0.00230249745158002,
          0.0011557084607543323, 1.783893985728848e-05, 0.0012003058103975538, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0034403669724770644, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881, 1.6864543982730707e-05,
          1.783893985728848e-05, 8.9194699286442398e-06, 0.0023114169215086642, 0.0011467889908256881,
          0.0011557084607543323, 2.5784013911374943e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881,
          0.0011557084607543323, 0.0011467889908256881, 8.9194699286442398e-06, 0.0011557084607543323,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 2.5784013911374943e-05, 1.6864543982730707e-05, 1.6864543982730707e-05,
          1.783893985728848e-05, 8.9194699286442398e-06, 0.0011557084607543323, 8.9194699286442398e-06,
          1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0024174761647778376,
          0.0012072764885770822, 0.010409321220843079, 0.0011467889908256881, 1.6864543982730707e-05,
          0.00230249745158002, 1.783893985728848e-05, 0.0011467889908256881, 0.0011467889908256881,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011557084607543323, 1.783893985728848e-05, 8.9194699286442398e-06,
          3.5677879714576959e-05, 8.9194699286442398e-06, 0.0011467889908256881, 3.5677879714576959e-05,
          0.0068985728848114181, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0034492864424057082,
          0.0022935779816513763, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.783893985728848e-05, 2.6758409785932719e-05, 8.9194699286442398e-06,
          0.0022935779816513763, 8.9194699286442398e-06, 3.5677879714576959e-05, 4.4597349643221202e-05,
          7.9300833483240382e-05, 1.783893985728848e-05, 0.0011636535348084186, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0068807339449541288, 8.9194699286442398e-06,
          8.9194699286442398e-06, 2.6758409785932719e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.018753073094681296, 0.00230249745158002, 0.001173547400611621,
          0.0011467889908256881, 0.0011557084607543323, 8.9194699286442398e-06, 0.0011557084607543323,
          8.9194699286442398e-06, 0.0022935779816513763, 0.001173547400611621, 1.6864543982730707e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 3.4703483840019187e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.6864543982730707e-05, 0.0011646279306829765, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.783893985728848e-05, 8.9194699286442398e-06, 0.0011646279306829765, 0.0011467889908256881,
          8.9194699286442398e-06, 4.4597349643221202e-05, 8.9194699286442398e-06, 0.0011824668705402652,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 2.6758409785932719e-05,
          0.0011467889908256881, 8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 3.5677879714576959e-05, 0.0011824668705402649,
          1.783893985728848e-05, 8.9194699286442398e-06, 0.001286577322060323, 0.0011467889908256881,
          8.9194699286442398e-06, 0.0049499310427534937, 0.0011814924746657075, 1.6864543982730707e-05,
          1.6864543982730707e-05, 0.0143632697727409, 5.2542423697307666e-05, 0.0082396414223181638,
          4.362295376866343e-05, 0.0011467889908256881, 0.0011557084607543323, 8.9194699286442398e-06,
          2.5784013911374943e-05, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 8.9194699286442398e-06,
          0.0011467889908256881, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 6.8432571805480594e-05, 0.0022935779816513763,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 0.0011557084607543323, 1.6864543982730707e-05, 1.783893985728848e-05,
          8.9194699286442398e-06, 0.0034929093961743717, 2.6758409785932719e-05, 8.9194699286442398e-06,
          1.783893985728848e-05, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0012181447502548422, 2.5784013911374943e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 6.2436289500509675e-05,
          0.0011646279306829768, 8.9194699286442398e-06, 0.0011646279306829768, 8.9194699286442398e-06,
          1.783893985728848e-05, 0.00230249745158002, 0.0023114169215086642, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.783893985728848e-05, 2.5784013911374943e-05, 4.362295376866343e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0046387239911255028, 8.9194699286442398e-06,
          0.02782132577801763, 8.9194699286442398e-06, 0.001173547400611621, 8.9194699286442398e-06,
          7.1355759429153918e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0046307789170714157, 0.0011467889908256881, 8.9194699286442398e-06,
          4.2648557894105657e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 2.6758409785932719e-05,
          8.9194699286442398e-06, 0.0011557084607543323, 0.0011557084607543323, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011557084607543323, 8.9194699286442398e-06, 0.0011467889908256881, 0.0011904119445943515,
          0.0011646279306829765, 0.00230249745158002, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 1.783893985728848e-05,
          1.6864543982730707e-05, 0.0022935779816513763, 1.783893985728848e-05, 8.9194699286442398e-06,
          1.6864543982730707e-05, 0.0011467889908256881, 1.783893985728848e-05, 8.9194699286442398e-06,
          0.0011467889908256881, 0.0011646279306829768, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 5.3516819571865445e-05, 2.6758409785932719e-05, 1.783893985728848e-05,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 0.0011467889908256881, 0.0011467889908256881, 8.9194699286442398e-06,
          1.6864543982730707e-05, 0.0011557084607543323, 0.0022935779816513763, 8.9194699286442398e-06,
          1.6864543982730707e-05, 0.0011467889908256881, 8.9194699286442398e-06, 2.6758409785932719e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.00230249745158002,
          8.9194699286442398e-06, 8.9194699286442398e-06, 7.7352041734124837e-05, 1.6864543982730707e-05,
          2.5784013911374943e-05, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881,
          0.0011467889908256881, 0.0035018288661030164, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 0.003511722731906218, 2.5784013911374943e-05, 0.0023470948012232419,
          1.6864543982730707e-05, 8.9194699286442398e-06, 0.0011824668705402649, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011557084607543323, 0.0034938837920489301, 0.0011725730047370631,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011646279306829765, 3.5677879714576959e-05,
          0.0012171703543802843, 2.6758409785932719e-05, 0.0011467889908256881, 0.0022935779816513763,
          0.001173547400611621, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011814924746657075,
          4.2648557894105657e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          3.4703483840019187e-05, 0.0011814924746657073, 0.0023114169215086642, 0.0011467889908256881,
          1.6864543982730707e-05, 8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06,
          1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.6864543982730707e-05,
          1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 0.0011467889908256881, 8.9194699286442398e-06, 1.783893985728848e-05,
          0.0011894375487197938, 0.0011904119445943515, 0.0011467889908256881, 0.0011467889908256881,
          1.6864543982730707e-05, 0.0011467889908256881, 0.0011467889908256881, 0.0023114169215086642,
          0.0011467889908256881, 2.5784013911374943e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.783893985728848e-05, 8.9194699286442398e-06, 0.0095232955567548117, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          2.6758409785932719e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.783893985728848e-05, 0.0034492864424057082, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0035008544702284585, 2.6758409785932719e-05,
          0.0012171703543802843, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881,
          1.783893985728848e-05, 2.5784013911374943e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 6.0487497751394144e-05, 0.00357318462553217, 1.783893985728848e-05,
          0.0058568687413803446, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.00230249745158002, 8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 2.6758409785932719e-05, 5.3516819571865445e-05,
          8.9194699286442398e-06, 0.0011467889908256881, 2.6758409785932719e-05, 0.0011467889908256881,
          0.015579315824189005, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          1.783893985728848e-05, 0.0011824668705402652, 8.9194699286442398e-06, 2.5784013911374943e-05,
          0.0011646279306829765, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 2.5784013911374943e-05, 1.6864543982730707e-05, 1.6864543982730707e-05,
          1.6864543982730707e-05, 0.0011467889908256881, 0.0011467889908256881, 2.6758409785932719e-05,
          8.9194699286442398e-06, 2.5784013911374943e-05, 1.6864543982730707e-05, 8.9194699286442398e-06,
          6.8432571805480594e-05, 0.0092446932901600991, 0.0011467889908256881, 2.5784013911374943e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.6864543982730707e-05,
          1.783893985728848e-05, 8.9194699286442398e-06, 0.0011805180787911494, 8.9194699286442398e-06,
          2.5784013911374943e-05, 8.9194699286442398e-06, 0.0069243568987227919, 0.002319361995562751,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0023292558613659535, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0011557084607543323, 0.0011467889908256881,
          8.9194699286442398e-06, 1.783893985728848e-05, 0.0011467889908256881, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0023114169215086642,
          0.0011467889908256881, 8.9194699286442398e-06, 0.0022935779816513763, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 1.783893985728848e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0012241410325598133, 0.0023104425256341068,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 2.6758409785932719e-05,
          8.9194699286442398e-06, 1.783893985728848e-05, 1.783893985728848e-05, 2.6758409785932719e-05,
          1.6864543982730707e-05, 0.0061661270012592197, 0.0011904119445943519, 8.9194699286442398e-06,
          8.9194699286442398e-06, 3.4703483840019187e-05, 4.2648557894105657e-05, 0.0011467889908256881,
          0.0011725730047370631, 0.0012330605024884575, 8.9194699286442398e-06, 0.0034572315164597949,
          0.0011467889908256881, 8.9194699286442398e-06, 2.6758409785932719e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011557084607543323, 0.0057954068477543931,
          8.9194699286442398e-06, 0.0023114169215086647, 8.9194699286442398e-06, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05, 0.001173547400611621,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 0.0011467889908256881,
          3.4703483840019187e-05, 1.6864543982730707e-05, 8.9194699286442398e-06, 1.6864543982730707e-05,
          0.0012063020927025247, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881,
          0.0011467889908256881, 0.0080979042993344127, 8.9194699286442398e-06, 3.4703483840019187e-05,
          8.9194699286442398e-06, 0.0011636535348084186, 3.4703483840019187e-05, 8.9194699286442398e-06,
          0.0011467889908256881, 1.783893985728848e-05, 1.6864543982730707e-05, 8.9194699286442398e-06,
          1.6864543982730707e-05, 6.1461893625951909e-05, 1.6864543982730707e-05, 8.9194699286442398e-06,
          1.783893985728848e-05, 0.0011814924746657073, 8.9194699286442398e-06, 1.783893985728848e-05,
          0.0011646279306829765, 1.783893985728848e-05, 8.9194699286442398e-06, 0.0011557084607543323,
          1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 1.783893985728848e-05, 1.783893985728848e-05, 1.783893985728848e-05,
          8.9194699286442398e-06, 1.783893985728848e-05, 0.0011646279306829765, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.00230249745158002, 0.0094727019248066202, 6.9406967680038373e-05, 0.0011725730047370631,
          1.6864543982730707e-05, 8.5297115788211315e-05, 8.9194699286442398e-06, 1.6864543982730707e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.00230249745158002, 6.9406967680038373e-05, 0.0011467889908256881, 0.0011467889908256881,
          8.9194699286442398e-06, 1.783893985728848e-05, 1.783893985728848e-05, 1.6864543982730707e-05,
          8.9194699286442398e-06, 0.0011467889908256881, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 1.783893985728848e-05,
          2.5784013911374943e-05, 8.9194699286442398e-06, 0.0023104425256341068, 2.5784013911374943e-05,
          1.6864543982730707e-05, 0.0011557084607543323, 8.9194699286442398e-06, 0.0011646279306829765,
          8.9194699286442398e-06, 2.6758409785932719e-05, 0.0011467889908256881, 8.9194699286442398e-06,
          1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.013020102536427415,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 3.4703483840019187e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05, 1.783893985728848e-05,
          3.3729087965461414e-05, 0.0011467889908256881, 0.0023203363914373089, 1.6864543982730707e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881,
          0.0011467889908256881, 0.0011735474006116212, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.783893985728848e-05, 2.6758409785932719e-05, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0012092252803261978, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0022935779816513763, 9.7139773340528869e-05,
          0.0011557084607543323, 0.0011467889908256881, 0.0011467889908256881, 0.0036158331834262756,
          0.0011467889908256881, 0.0011467889908256881, 8.9194699286442398e-06, 0.0058727588894885172,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0034403669724770644, 0.0011467889908256881,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          1.783893985728848e-05, 2.6758409785932719e-05, 0.0034403669724770644, 2.5784013911374943e-05,
          3.4703483840019187e-05, 0.0011557084607543323, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06, 0.0022935779816513763,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05, 0.0011557084607543323,
          1.6864543982730707e-05, 0.0011467889908256881, 0.0045871559633027525, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 2.5784013911374943e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881, 0.0011557084607543323,
          0.0011467889908256881, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881,
          0.0011467889908256881, 8.9194699286442398e-06, 1.6864543982730707e-05, 0.00230249745158002,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011824668705402652,
          5.0593631948192115e-05, 8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06,
          2.5784013911374943e-05, 0.0011636535348084186, 2.6758409785932719e-05, 0.0034839899262457275,
          0.0011467889908256881, 1.6864543982730707e-05, 8.9194699286442398e-06, 1.6864543982730707e-05,
          1.783893985728848e-05, 8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06,
          0.001209225280326198, 8.9194699286442398e-06, 1.6864543982730707e-05, 0.0011467889908256881,
          3.5677879714576959e-05, 8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06,
          0.0011467889908256881, 0.0011467889908256881, 8.9194699286442398e-06, 0.0011557084607543323,
          0.0046654824009114354, 8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06,
          3.3729087965461414e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          5.2542423697307666e-05, 0.00230249745158002, 8.9194699286442398e-06, 0.0011557084607543323,
          8.9194699286442398e-06, 1.6864543982730707e-05, 8.9194699286442398e-06, 0.0012082508844516403,
          1.783893985728848e-05, 0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06,
          1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 0.0011467889908256881, 0.0022935779816513763,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          2.6758409785932719e-05, 0.0011557084607543323, 4.2648557894105657e-05, 8.9194699286442398e-06,
          0.0011467889908256881, 0.003511722731906218, 0.0011467889908256881, 8.9194699286442398e-06,
          8.9194699286442398e-06, 2.5784013911374943e-05, 1.783893985728848e-05, 8.9194699286442398e-06,
          1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          0.0011725730047370631, 1.783893985728848e-05, 2.6758409785932719e-05, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 2.6758409785932719e-05, 8.9194699286442398e-06,
          2.6758409785932719e-05, 0.0011557084607543323, 4.2648557894105657e-05, 1.783893985728848e-05,
          8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06, 0.0011467889908256881,
          1.6864543982730707e-05, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0024066079031000781,
          8.9194699286442398e-06, 0.0011467889908256881, 0.0012449031600407748, 0.0095331894225580156,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 8.9194699286442398e-06, 1.6864543982730707e-05, 0.0011824668705402649,
          8.9194699286442398e-06, 2.6758409785932719e-05, 1.6864543982730707e-05, 8.9194699286442398e-06,
          0.0011725730047370631, 0.0011557084607543323, 0.0011467889908256881, 0.0023461204053486836,
          3.4703483840019187e-05, 1.6864543982730707e-05, 0.0011557084607543323, 1.783893985728848e-05,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 5.3516819571865439e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881,
          0.0057954068477543931, 0.0011467889908256881, 8.9194699286442398e-06, 2.5784013911374943e-05,
          0.0023114169215086647, 8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 8.9194699286442398e-06, 0.0011467889908256881, 0.0011467889908256881,
          0.0011646279306829765, 0.0011557084607543323, 0.015137239911255025, 0.0023649337410805299,
          0.0011467889908256881, 0.0058667626071835458, 0.0011646279306829768, 0.0011467889908256881,
          7.1355759429153918e-05, 0.0011557084607543323, 0.0011646279306829768, 2.6758409785932719e-05,
          8.9194699286442398e-06, 8.9194699286442398e-06, 1.783893985728848e-05, 8.9194699286442398e-06,
          0.0011467889908256881, 8.9194699286442398e-06, 0.0011557084607543323, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.002319361995562751, 5.3516819571865445e-05, 0.0011646279306829768,
          0.0011646279306829768, 0.017940501888828928, 8.9194699286442398e-06, 0.007258424776638483,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.6864543982730707e-05,
          0.002319361995562751, 8.9194699286442398e-06, 0.0095252443485039275, 0.0012082508844516401,
          8.9194699286442398e-06, 0.00013379204892966361, 8.9194699286442398e-06, 8.9194699286442398e-06,
          1.6864543982730707e-05, 8.9194699286442398e-06, 1.6864543982730707e-05, 2.6758409785932719e-05,
          2.5784013911374943e-05, 2.5784013911374943e-05, 1.783893985728848e-05, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011824668705402652, 3.5677879714576959e-05,
          8.9194699286442398e-06, 2.5784013911374943e-05, 8.9194699286442398e-06, 8.9194699286442398e-06,
          8.9194699286442398e-06, 0.0012181447502548422, 0.0012538226299694188, 8.9194699286442398e-06,
          8.9194699286442398e-06, 8.9194699286442398e-06, 8.9194699286442398e-06, 1.6864543982730707e-05,
          0.0011467889908256881, 8.9194699286442398e-06, 8.9194699286442398e-06, 0.0011467889908256881,
          7.8326437608682603e-05, 1.6864543982730707e-05, 8.9194699286442398e-06, 0.0036177819751753914,
          2.6758409785932719e-05, 8.9194699286442398e-06, 0.004771541644180608, 8.9194699286442398e-06,
          1.783893985728848e-05, 1.783893985728848e-05, 8.9194699286442398e-06, 0.0082020147508544694,
          0.0023451460094741261, 0.00015860166696648076, 0.023629699586256523, 0.0012241410325598131]]
    for i in range(1, len(a)):
        for j in range(1, len(a[i])):
            a[i][j] *= 900
    wordlists = matrixtodict(a)
    print 'ge', wordlists[0]['ge'], 'wisdom', wordlists[0]['wisdom'], 'swefnes', wordlists[0]['swefnes']
    print

    print sort(testall(wordlists))
    print
    lalala = groupdivision(wordlists, [[0], [1, 2, 3, 4]])
    temp = testgroup(lalala)
    for key in temp.keys():
        print key, temp[key][:10]

    print
    lalala = groupdivision(wordlists, [[2], [1, 0, 3, 4]])
    temp = testgroup(lalala)
    for key in temp.keys():
        print key, temp[key][:10]

    print
    print 'no grey word'
    a = [['', 'a', 'aban', 'abead', 'abrahame', 'abrahames', 'abrecan', 'ac', 'aceorfe\xc3\xb0', 'acol', 'acul', 'acw\xc3\xa6\xc3\xb0', 'adzarias', 'afeallan', 'aferan', 'agan', 'agangen', 'aglac', 'ag\xc3\xa6f', 'ahicgan', 'ahte', 'ahton', 'ahwearf', 'aldor', 'aldordom', 'aldordomes', 'aldorlege', 'aldre', 'alet', 'alhstede', 'alwihta', 'alysde', 'al\xc3\xa6t', 'al\xc3\xa6ten', 'an', 'ana', 'and', 'andan', 'andsaca', 'andswaredon', 'andswarode', 'ane', 'angin', 'anhydig', 'anmedlan', 'anmod', 'annanias', 'anne', 'anra', 'anwalh', 'ar', 'are', 'areccan', 'arehte', 'arna', 'ar\xc3\xa6dan', 'ar\xc3\xa6dde', 'ar\xc3\xa6rde', 'asecganne', 'asette', 'astah', 'astige\xc3\xb0', 'as\xc3\xa6gde', 'ateah', 'awacodon', 'awoc', 'aworpe', 'awunnen', 'azarias', 'a\xc3\xbeencean', 'babilon', 'babilone', 'babilonia', 'babilonie', 'babilonige', 'baldazar', 'balde', 'banum', 'baswe', 'be', 'beacen', 'beacne', 'bead', 'beam', 'beames', 'bearn', 'bearnum', 'bearwe', 'bebead', 'bebodes', 'bebodo', 'bebuga\xc3\xb0', 'becwom', 'befolen', 'begete', 'belegde', 'belocene', 'bende', 'beon', 'beorgas', 'beorht', 'beorhte', 'beorn', 'beornas', 'beot', 'beote', 'beran', 'bera\xc3\xb0', 'bere', 'berhtmhwate', 'beseah', 'besn\xc3\xa6dan', 'besn\xc3\xa6ded', 'beswac', 'besw\xc3\xa6led', 'beteran', 'bewindan', 'bewr\xc3\xa6con', 'be\xc3\xbeeahte', 'bidda\xc3\xb0', 'billa', 'bitera', 'bi\xc3\xb0', 'blacan', 'blace', 'bleda', 'bledum', 'bletsian', 'bletsia\xc3\xb0', 'bletsie', 'bletsige', 'blican', 'bli\xc3\xb0e', 'bli\xc3\xb0emod', 'bli\xc3\xb0emode', 'bl\xc3\xa6d', 'bl\xc3\xa6de', 'bl\xc3\xa6dum', 'boca', 'bocerum', 'bocstafas', 'bolgenmod', 'bote', 'bradne', 'brandas', 'brego', 'brema\xc3\xb0', 'breme', 'breostge\xc3\xb0ancum', 'breostlocan', 'bresne', 'brimfaro\xc3\xbees', 'brohte', 'brungen', 'bryne', 'brytnedon', 'bryttedon', 'br\xc3\xa6con', 'br\xc3\xa6sna', 'bude', 'bun', 'burga', 'burge', 'burh', 'burhge', 'burhsittende', 'burhsittendum', 'burhware', 'burnon', 'butan', 'bylywit', 'byman', 'byrig', 'byrnende', 'b\xc3\xa6d', 'b\xc3\xa6don', 'b\xc3\xa6lblyse', 'b\xc3\xa6le', 'b\xc3\xa6r', 'b\xc3\xa6rnan', 'b\xc3\xa6ron', 'caldea', 'caldeas', 'can', 'ceald', 'ceapian', 'ceastergeweorc', 'ceastre', 'cempan', 'cenned', 'ceorfan', 'clammum', 'cl\xc3\xa6ne', 'cneomagum', 'cneorissum', 'cneow', 'cneowum', 'cnihta', 'cnihtas', 'cnihton', 'cnihtum', 'com', 'come', 'comon', 'cor\xc3\xb0res', 'cr\xc3\xa6ft', 'cr\xc3\xa6ftas', 'cuman', 'cumble', 'cunnode', 'cunnon', 'curon', 'cu\xc3\xb0', 'cu\xc3\xb0on', 'cu\xc3\xb0ost', 'cwale', 'cwealme', 'cwelm', 'cwe\xc3\xb0an', 'cwe\xc3\xb0a\xc3\xb0', 'cwom', 'cwome', 'cw\xc3\xa6don', 'cw\xc3\xa6\xc3\xb0', 'cyme', 'cymst', 'cyn', 'cynegode', 'cyne\xc3\xb0rymme', 'cynig', 'cyning', 'cyningdom', 'cyningdome', 'cyninge', 'cyninges', 'cyrdon', 'cyst', 'cy\xc3\xb0an', 'daga', 'daniel', 'deaw', 'dea\xc3\xb0', 'dea\xc3\xb0e', 'dema', 'deoflu', 'deoflum', 'deofolwitgan', 'deopne', 'deor', 'deora', 'deormode', 'deorum', 'derede', 'de\xc3\xb0', 'diran', 'dom', 'domas', 'dome', 'domige', 'don', 'dreag', 'dreamas', 'dreame', 'dreamleas', 'drearung', 'drihten', 'drihtenweard', 'drihtne', 'drihtnes', 'drincan', 'dropena', 'drugon', 'dryge', 'duge\xc3\xb0e', 'duge\xc3\xbeum', 'dugu\xc3\xb0e', 'dyde', 'dydon', 'dyglan', 'dygle', 'd\xc3\xa6da', 'd\xc3\xa6de', 'd\xc3\xa6dhwatan', 'd\xc3\xa6g', 'd\xc3\xa6ge', 'd\xc3\xa6ges', 'eac', 'eacenne', 'eacne', 'ead', 'eadmodum', 'eagum', 'ealdfeondum', 'ealdor', 'ealdormen', 'ealhstede', 'eall', 'ealle', 'ealles', 'eallum', 'ealne', 'ealra', 'earce', 'eard', 'eare', 'earfo\xc3\xb0m\xc3\xa6cg', 'earfo\xc3\xb0si\xc3\xb0as', 'earme', 'earmra', 'earmre', 'earmsceapen', 'eart', 'eastream', 'ea\xc3\xb0medum', 'ebrea', 'ece', 'ecgum', 'ecne', 'edsceafte', 'efnde', 'efndon', 'efne', 'eft', 'egesa', 'egesan', 'egesful', 'egeslic', 'egeslicu', 'egle', 'ehtode', 'ende', 'ended\xc3\xa6g', 'endelean', 'engel', 'englas', 'engles', 'eode', 'eodon', 'eorla', 'eorlas', 'eorlum', 'eor\xc3\xb0an', 'eor\xc3\xb0buendum', 'eor\xc3\xb0cyninga', 'eor\xc3\xb0lic', 'eowed', 'eower', 'esnas', 'est', 'e\xc3\xb0el', 'facne', 'fandedon', 'fea', 'feax', 'fela', 'feld', 'felda', 'feohsceattum', 'feonda', 'feondas', 'feore', 'feorh', 'feorhnere', 'feorum', 'feor\xc3\xb0a', 'feower', 'feran', 'findan', 'fleam', 'fleon', 'folc', 'folca', 'folce', 'folcgesi\xc3\xb0um', 'folcm\xc3\xa6gen', 'folctoga', 'folctogan', 'folcum', 'foldan', 'for', 'foran', 'forbr\xc3\xa6con', 'forburnene', 'foremihtig', 'forfangen', 'forgeaf', 'forht', 'forh\xc3\xa6fed', 'forlet', 'forstas', 'for\xc3\xb0am', 'for\xc3\xb0on', 'for\xc3\xbeam', 'fraco\xc3\xb0', 'fram', 'frasade', 'frea', 'freagleawe', 'frean', 'frecnan', 'frecne', 'fremde', 'fremede', 'freobearn', 'freo\xc3\xb0o', 'fri\xc3\xb0', 'fri\xc3\xb0e', 'fri\xc3\xb0es', 'frod', 'frofre', 'frumcyn', 'frumgaras', 'frumsl\xc3\xa6pe', 'frumspr\xc3\xa6ce', 'fr\xc3\xa6gn', 'fr\xc3\xa6twe', 'fuglas', 'fugolas', 'funde', 'fundon', 'fur\xc3\xb0or', 'fyl', 'fyll', 'fyr', 'fyre', 'fyrenan', 'fyrend\xc3\xa6dum', 'fyrene', 'fyrenum', 'fyres', 'fyrndagum', 'fyrstmearc', 'f\xc3\xa6c', 'f\xc3\xa6der', 'f\xc3\xa6gre', 'f\xc3\xa6r', 'f\xc3\xa6rgryre', 'f\xc3\xa6st', 'f\xc3\xa6stan', 'f\xc3\xa6ste', 'f\xc3\xa6stlicne', 'f\xc3\xa6stna', 'f\xc3\xa6stne', 'f\xc3\xa6\xc3\xb0m', 'f\xc3\xa6\xc3\xb0me', 'f\xc3\xa6\xc3\xb0mum', 'gad', 'gang', 'gangan', 'gange', 'gast', 'gasta', 'gastas', 'gaste', 'gastes', 'gastum', 'ge', 'gealhmod', 'gealp', 'gearo', 'gearu', 'gebead', 'gebearh', 'gebede', 'gebedu', 'gebedum', 'gebindan', 'gebletsad', 'gebletsige', 'geboden', 'geborgen', 'geb\xc3\xa6don', 'gecoren', 'gecorene', 'gecw\xc3\xa6don', 'gecw\xc3\xa6\xc3\xb0', 'gecy\xc3\xb0', 'gecy\xc3\xb0de', 'gecy\xc3\xb0ed', 'gedemed', 'gedon', 'gedydon', 'geegled', 'geflymed', 'gefrecnod', 'gefremede', 'gefrigen', 'gefrunon', 'gefr\xc3\xa6ge', 'gefr\xc3\xa6gn', 'gef\xc3\xa6gon', 'gegleded', 'gegnunga', 'gehete', 'gehogode', 'gehwam', 'gehwearf', 'gehwilc', 'gehwilcum', 'gehwurfe', 'gehw\xc3\xa6s', 'gehydum', 'gehyge', 'gehyrdon', 'geleafan', 'gelic', 'gelicost', 'gelimpan', 'gelyfan', 'gelyfde', 'gelyfest', 'gel\xc3\xa6dde', 'gel\xc3\xa6ste', 'gemenged', 'gemet', 'gemunan', 'gemunde', 'gemynd', 'gemyndgast', 'gem\xc3\xa6ne', 'gem\xc3\xa6ted', 'gem\xc3\xa6tte', 'genamon', 'generede', 'gengum', 'genumen', 'geoca', 'geoce', 'geocre', 'geocrostne', 'geogo\xc3\xb0e', 'geond', 'geondsawen', 'geonge', 'georn', 'georne', 'gerefan', 'gerume', 'gerusalem', 'gerynu', 'gerysna', 'ger\xc3\xa6dum', 'gesawe', 'gesawon', 'gesceaft', 'gesceafta', 'gesceafte', 'gesceod', 'gesceode', 'gescylde', 'geseah', 'geseald', 'gesecganne', 'geseo', 'geseted', 'gese\xc3\xb0ed', 'gesigef\xc3\xa6ste', 'gesi\xc3\xb0', 'gesloh', 'gespr\xc3\xa6c', 'gestreon', 'geswi\xc3\xb0de', 'gesyh\xc3\xb0e', 'ges\xc3\xa6de', 'ges\xc3\xa6ledne', 'ges\xc3\xa6t', 'getenge', 'geteod', 'geteode', 'gewand', 'gewat', 'geweald', 'gewealde', 'gewear\xc3\xb0', 'gewemman', 'gewemmed', 'geweox', 'gewindagum', 'gewit', 'gewita', 'gewittes', 'geworden', 'gewordene', 'geworhte', 'gewur\xc3\xb0ad', 'gewur\xc3\xb0od', 'gewyrhto', 'ge\xc3\xb0afian', 'ge\xc3\xb0anc', 'ge\xc3\xb0ances', 'ge\xc3\xb0ancum', 'ge\xc3\xb0enc', 'ge\xc3\xb0inges', 'ge\xc3\xbeanc', 'ge\xc3\xbeeahte', 'ge\xc3\xbeingu', 'gif', 'gife', 'gifena', 'ginge', 'gingum', 'glade', 'gleaw', 'gleawmode', 'gleawost', 'gleda', 'gl\xc3\xa6dmode', 'god', 'gode', 'godes', 'godspellode', 'gods\xc3\xa6de', 'gold', 'golde', 'goldfatu', 'gramlice', 'grene', 'grim', 'grimman', 'grimme', 'grimmost', 'grome', 'grund', 'grynde\xc3\xb0', 'gryre', 'gr\xc3\xa6s', 'gulpon', 'guman', 'gumena', 'gumrices', 'gumum', 'gyddedon', 'gyddigan', 'gyfe', 'gyfum', 'gyld', 'gyldan', 'gylde', 'gyldnan', 'gylp', 'gylpe', 'g\xc3\xa6delingum', 'g\xc3\xa6st', 'habban', 'habba\xc3\xb0', 'had', 'hade', 'hale', 'halegu', 'halga', 'halgan', 'halgum', 'halig', 'halige', 'haliges', 'haligra', 'haligu', 'hamsittende', 'hand', 'hat', 'hata', 'hatan', 'haten', 'hatne', 'hatte', 'he', 'hea', 'heah', 'heahbyrig', 'heahcyning', 'heahheort', 'healdan', 'healda\xc3\xb0', 'healle', 'hean', 'heane', 'heanne', 'heapum', 'hearan', 'hearde', 'hearm', 'hebbanne', 'hefonfugolas', 'hegan', 'heh', 'help', 'helpe', 'helpend', 'heofenum', 'heofnum', 'heofona', 'heofonas', 'heofonbeorht', 'heofones', 'heofonheane', 'heofonrices', 'heofonsteorran', 'heofontunglum', 'heofonum', 'heold', 'heolde', 'heora', 'heorta', 'heortan', 'heorugrimra', 'here', 'herede', 'heredon', 'herega', 'heretyma', 'herewosan', 'hergas', 'herga\xc3\xb0', 'hergende', 'heriga\xc3\xb0', 'herige', 'heriges', 'herran', 'het', 'hete', 'heton', 'hie', 'hige', 'higecr\xc3\xa6ft', 'hige\xc3\xbeancle', 'him', 'hine', 'his', 'hit', 'hlaford', 'hleo', 'hleo\xc3\xb0or', 'hleo\xc3\xb0orcwyde', 'hleo\xc3\xb0orcyme', 'hleo\xc3\xb0rade', 'hlifigan', 'hlifode', 'hliga\xc3\xb0', 'hluttor', 'hlypum', 'hlyst', 'hofe', 'hogedon', 'hold', 'holt', 'hordm\xc3\xa6gen', 'horsce', 'hra\xc3\xb0e', 'hra\xc3\xb0or', 'hreddan', 'hremde', 'hreohmod', 'hre\xc3\xb0', 'hrof', 'hrofe', 'hrusan', 'hryre', 'hr\xc3\xa6gle', 'hu', 'huslfatu', 'hwa', 'hwalas', 'hwearf', 'hweorfan', 'hwilc', 'hwile', 'hwurfan', 'hwurfon', 'hwyrft', 'hw\xc3\xa6t', 'hw\xc3\xa6\xc3\xb0ere', 'hw\xc3\xa6\xc3\xb0re', 'hyge', 'hyld', 'hyldelease', 'hyldo', 'hyllas', 'hyra', 'hyran', 'hyrde', 'hyrdon', 'hyrra', 'hyrran', 'hyssas', 'h\xc3\xa6fde', 'h\xc3\xa6fdest', 'h\xc3\xa6fdon', 'h\xc3\xa6ft', 'h\xc3\xa6ftas', 'h\xc3\xa6le\xc3\xb0', 'h\xc3\xa6le\xc3\xb0a', 'h\xc3\xa6le\xc3\xb0um', 'h\xc3\xa6to', 'h\xc3\xa6\xc3\xb0en', 'h\xc3\xa6\xc3\xb0ena', 'h\xc3\xa6\xc3\xb0enan', 'h\xc3\xa6\xc3\xb0endom', 'h\xc3\xa6\xc3\xb0ene', 'h\xc3\xa6\xc3\xb0enra', 'h\xc3\xa6\xc3\xb0ne', 'h\xc3\xa6\xc3\xb0num', 'iacobe', 'ic', 'ican', 'in', 'inge\xc3\xbeancum', 'innan', 'inne', 'is', 'isaace', 'isen', 'iserne', 'isernum', 'israela', 'iudea', 'lacende', 'lafe', 'lagon', 'lagostreamas', 'landa', 'landgesceaft', 'lange', 'lare', 'larum', 'la\xc3\xb0', 'la\xc3\xb0e', 'la\xc3\xb0searo', 'lean', 'leas', 'leng', 'lengde', 'leoda', 'leode', 'leodum', 'leofum', 'leoge\xc3\xb0', 'leoht', 'leohtfruma', 'leohtran', 'leoman', 'leornedon', 'let', 'lice', 'lif', 'lifde', 'life', 'lifes', 'liffrean', 'liffruman', 'lifgende', 'lifigea\xc3\xb0', 'lifigen', 'lifigende', 'lig', 'lige', 'liges', 'ligetu', 'ligeword', 'ligges', 'lignest', 'lisse', 'li\xc3\xb0', 'locia\xc3\xb0', 'locode', 'lof', 'lofia\xc3\xb0', 'lofige', 'lufan', 'lufia\xc3\xb0', 'lust', 'lyfte', 'lyftlacende', 'lyhte', 'lytel', 'l\xc3\xa6g', 'ma', 'magon', 'man', 'mancynne', 'mandreame', 'mandrihten', 'mandrihtne', 'mane', 'manegum', 'manig', 'manlican', 'manna', 'mannum', 'mara', 'mare', 'me', 'meaht', 'meahte', 'meda', 'medugal', 'medum', 'meld', 'men', 'menigo', 'merestreamas', 'meted', 'metod', 'metode', 'metodes', 'me\xc3\xb0elstede', 'me\xc3\xb0le', 'micel', 'micelne', 'miclan', 'micle', 'mid', 'middangeard', 'middangeardes', 'migtigra', 'miht', 'mihta', 'mihte', 'mihtig', 'mihtigran', 'mihton', 'mihtum', 'miltse', 'miltsum', 'min', 'mine', 'minra', 'minsode', 'mirce', 'misael', 'mod', 'mode', 'modge\xc3\xb0anc', 'modge\xc3\xbeances', 'modhwatan', 'modig', 'modsefan', 'modum', 'moldan', 'mona', 'monig', 'monige', 'mores', 'mor\xc3\xb0re', 'moste', 'myndga\xc3\xb0', 'm\xc3\xa6', 'm\xc3\xa6cgum', 'm\xc3\xa6ge', 'm\xc3\xa6gen', 'm\xc3\xa6genes', 'm\xc3\xa6lmete', 'm\xc3\xa6nige', 'm\xc3\xa6nigeo', 'm\xc3\xa6re', 'm\xc3\xa6rost', 'm\xc3\xa6st', 'm\xc3\xa6tinge', 'm\xc3\xa6tra', 'na', 'nabochodonossor', 'nacod', 'nales', 'nalles', 'nama', 'naman', 'name', 'ne', 'neata', 'neh', 'nehstum', 'neod', 'nerede', 'nergend', 'nergenne', 'nerigende', 'niht', 'nis', 'ni\xc3\xb0', 'ni\xc3\xb0a', 'ni\xc3\xb0as', 'ni\xc3\xb0hete', 'ni\xc3\xb0wracum', 'no', 'noldon', 'nu', 'nydde', 'nyde', 'nydgenga', 'nym\xc3\xb0e', 'nym\xc3\xbee', 'ny\xc3\xb0or', 'n\xc3\xa6nig', 'n\xc3\xa6ron', 'n\xc3\xa6s', 'of', 'ofen', 'ofer', 'oferfaren', 'oferf\xc3\xa6\xc3\xb0mde', 'oferhogedon', 'oferhyd', 'oferhygd', 'oferhygde', 'oferhygdum', 'ofermedlan', 'ofestum', 'ofn', 'ofne', 'ofnes', 'ofstlice', 'oft', 'oftor', 'on', 'oncw\xc3\xa6\xc3\xb0', 'onegdon', 'onfenge', 'onfon', 'ongan', 'ongeald', 'ongeat', 'onget', 'onginnan', 'ongunnon', 'ongyt', 'onhicga\xc3\xb0', 'onhnigon', 'onhwearf', 'onhweorfe\xc3\xb0', 'onh\xc3\xa6tan', 'onh\xc3\xa6ted', 'onlah', 'onm\xc3\xa6lde', 'onsended', 'onsoce', 'onsocon', 'onsteallan', 'ontreowde', 'onwoc', 'or', 'ord', 'ordfruma', 'orlegra', 'orl\xc3\xa6g', 'owiht', 'owihtes', 'o\xc3\xb0', 'o\xc3\xb0er', 'o\xc3\xb0stod', 'o\xc3\xb0\xc3\xb0e', 'o\xc3\xb0\xc3\xbe\xc3\xa6t', 'persum', 'reccan', 'reccend', 'regna', 'rehte', 'reordberend', 'reorde', 'rest', 'reste', 'restende', 're\xc3\xb0e', 'rica', 'rice', 'rices', 'riht', 'rihte', 'rihtne', 'rodera', 'roderum', 'rodora', 'rodorbeorhtan', 'rodore', 'rohton', 'rume', 'run', 'runcr\xc3\xa6ftige', 'ryne', 'r\xc3\xa6d', 'r\xc3\xa6dan', 'r\xc3\xa6das', 'r\xc3\xa6df\xc3\xa6st', 'r\xc3\xa6dleas', 'r\xc3\xa6rde', 'r\xc3\xa6swa', 'salomanes', 'samnode', 'samod', 'sand', 'sawla', 'sawle', 'sceal', 'scealcas', 'sceatas', 'sceod', 'sceolde', 'sceoldon', 'scima', 'scine\xc3\xb0', 'scufan', 'scur', 'scyde', 'scylde', 'scyldig', 'scyppend', 'scyrede', 'se', 'sealde', 'sealte', 'sealtne', 'secan', 'secgan', 'secge', 'sefa', 'sefan', 'sel', 'seld', 'sele', 'sellende', 'sende', 'sended', 'sende\xc3\xb0', 'sendon', 'sennera', 'seo', 'seofan', 'seofon', 'seolfes', 'seon', 'septon', 'settend', 'sidne', 'sie', 'sien', 'siendon', 'sigora', 'sin', 'sine', 'sines', 'sinne', 'sinra', 'sinum', 'si\xc3\xb0', 'si\xc3\xb0estan', 'si\xc3\xb0f\xc3\xa6t', 'si\xc3\xb0ian', 'si\xc3\xb0\xc3\xb0an', 'sloh', 'sl\xc3\xa6pe', 'snawas', 'snotor', 'snytro', 'snyttro', 'sohton', 'somod', 'sona', 'sorge', 'sorh', 'so\xc3\xb0', 'so\xc3\xb0an', 'so\xc3\xb0cwidum', 'so\xc3\xb0e', 'so\xc3\xb0f\xc3\xa6st', 'so\xc3\xb0f\xc3\xa6stra', 'so\xc3\xb0ra', 'so\xc3\xb0um', 'sped', 'spel', 'spelboda', 'spelbodan', 'spowende', 'spreca\xc3\xb0', 'spr\xc3\xa6c', 'starude', 'sta\xc3\xb0ole', 'stefn', 'stefne', 'stigan', 'stille', 'stod', 'stode', 'strudon', 'sum', 'sumera', 'sumeres', 'sumor', 'sundor', 'sundorgife', 'sungon', 'sunna', 'sunnan', 'sunne', 'sunu', 'susl', 'swa', 'swefen', 'swefn', 'swefnede', 'swefnes', 'sweg', 'swelta\xc3\xb0', 'swigode', 'swilce', 'swi\xc3\xb0', 'swi\xc3\xb0an', 'swi\xc3\xb0e', 'swi\xc3\xb0mod', 'swi\xc3\xb0rian', 'swi\xc3\xb0rode', 'swutol', 'swylc', 'swylce', 'sw\xc3\xa6f', 'syle', 'sylf', 'sylfa', 'sylfe', 'sylle', 'symble', 'syndon', 's\xc3\xa6de', 's\xc3\xa6don', 's\xc3\xa6faro\xc3\xb0a', 's\xc3\xa6gde', 's\xc3\xa6gdon', 's\xc3\xa6t', 's\xc3\xa6ton', 's\xc3\xa6w\xc3\xa6gas', 'tacen', 'tacna', 'telgum', 'tempel', 'teode', 'teodest', 'teonfullum', 'teso', 'tid', 'tida', 'tide', 'tirum', 'to', 'todrifen', 'todw\xc3\xa6sced', 'tohworfene', 'torhtan', 'tosceaf', 'tosomne', 'toswende', 'tosweop', 'towrecene', 'treddedon', 'treow', 'treowum', 'trymede', 'tunglu', 'twigum', 'ufan', 'unbli\xc3\xb0e', 'unceapunga', 'under', 'ungelic', 'ungescead', 'unlytel', 'unriht', 'unrihtdom', 'unrihtum', 'unrim', 'unr\xc3\xa6d', 'unscyndne', 'unwaclice', 'up', 'upcyme', 'uppe', 'us', 'user', 'usic', 'ut', 'utan', 'wage', 'waldend', 'wall', 'wandode', 'wast', 'wa\xc3\xb0e', 'we', 'wealde\xc3\xb0', 'wealla', 'wealle', 'weard', 'weardas', 'weardode', 'wearmlic', 'wear\xc3\xb0', 'wece\xc3\xb0', 'wecga\xc3\xb0', 'weder', 'wedera', 'wedere', 'weg', 'welan', 'wendan', 'wende', 'weoh', 'weold', 'weorca', 'weor\xc3\xb0e\xc3\xb0', 'wer', 'wera', 'weras', 'wereda', 'werede', 'weredes', 'werigra', 'weroda', 'werode', 'werodes', 'werum', 'wer\xc3\xb0eode', 'wes', 'wesan', 'westen', 'wiccungdom', 'widan', 'wide', 'wideferh\xc3\xb0', 'widne', 'widost', 'wig', 'wiges', 'wihgyld', 'wiht', 'wihte', 'wildan', 'wilddeor', 'wilddeora', 'wilddeorum', 'wildeora', 'wildra', 'wildu', 'willa', 'willan', 'willa\xc3\xb0', 'wille', 'wilnedan', 'wilnian', 'winburge', 'winde', 'windig', 'windruncen', 'wine', 'wineleasne', 'wingal', 'winter', 'winterbiter', 'wintra', 'wis', 'wisa', 'wisdom', 'wise', 'wislice', 'wisne', 'wisse', 'wiste', 'wiston', 'wite', 'witegena', 'witga', 'witgode', 'witgum', 'witig', 'witiga\xc3\xb0', 'witigdom', 'witod', 'wi\xc3\xb0', 'wi\xc3\xb0erbreca', 'wlancan', 'wlenco', 'wlite', 'wlitescyne', 'wlitig', 'wlitiga', 'wod', 'wodan', 'wolcenfaru', 'wolcna', 'wolde', 'wolden', 'woldon', 'wom', 'woma', 'woman', 'womma', 'worce', 'word', 'worda', 'wordcwide', 'wordcwyde', 'worde', 'worden', 'wordgleaw', 'wordum', 'worhton', 'world', 'worlde', 'worn', 'woruld', 'woruldcr\xc3\xa6fta', 'worulde', 'woruldgesceafta', 'woruldlife', 'woruldrice', 'woruldspedum', 'wrace', 'wrat', 'wrece\xc3\xb0', 'writan', 'write', 'wroht', 'wr\xc3\xa6c', 'wr\xc3\xa6cca', 'wr\xc3\xa6clic', 'wr\xc3\xa6stran', 'wudu', 'wudubeam', 'wudubeames', 'wuldor', 'wuldorcyning', 'wuldorf\xc3\xa6st', 'wuldorhaman', 'wuldre', 'wuldres', 'wulfheort', 'wunast', 'wunden', 'wundor', 'wundorlic', 'wundra', 'wundre', 'wundrum', 'wunian', 'wunia\xc3\xb0', 'wunode', 'wurde', 'wurdon', 'wurpon', 'wur\xc3\xb0an', 'wur\xc3\xb0edon', 'wur\xc3\xb0ia\xc3\xb0', 'wur\xc3\xb0igean', 'wur\xc3\xb0myndum', 'wylla', 'wylm', 'wynsum', 'wyrcan', 'wyrd', 'wyrda', 'wyrrestan', 'wyrtruma', 'wyrtruman', 'wyrtum', 'w\xc3\xa6da', 'w\xc3\xa6de', 'w\xc3\xa6fran', 'w\xc3\xa6g', 'w\xc3\xa6re', 'w\xc3\xa6rf\xc3\xa6ste', 'w\xc3\xa6rgenga', 'w\xc3\xa6ron', 'w\xc3\xa6s', 'w\xc3\xa6ter', 'w\xc3\xa6terscipe', 'w\xc3\xa6tersprync', 'yfel', 'ylda', 'yldran', 'yldum', 'ymb', 'ymbe', 'yrre', 'ywed', 'y\xc3\xb0a', '\xc3\xa6', '\xc3\xa6cr\xc3\xa6ftig', '\xc3\xa6fre', '\xc3\xa6fter', '\xc3\xa6f\xc3\xa6ste', '\xc3\xa6ghw\xc3\xa6s', '\xc3\xa6ht', '\xc3\xa6hta', '\xc3\xa6hte', '\xc3\xa6lbeorht', '\xc3\xa6led', '\xc3\xa6lmihtig', '\xc3\xa6lmihtiges', '\xc3\xa6lmihtigne', '\xc3\xa6lmyssan', '\xc3\xa6nig', '\xc3\xa6r', '\xc3\xa6rendbec', '\xc3\xa6renum', '\xc3\xa6rest', '\xc3\xa6t', '\xc3\xa6tb\xc3\xa6r', '\xc3\xa6te', '\xc3\xa6tywed', '\xc3\xa6\xc3\xb0ele', '\xc3\xa6\xc3\xb0eling', '\xc3\xa6\xc3\xb0elinga', '\xc3\xa6\xc3\xb0elingas', '\xc3\xa6\xc3\xb0elinge', '\xc3\xa6\xc3\xb0elum', '\xc3\xb0a', '\xc3\xb0am', '\xc3\xb0ara', '\xc3\xb0e', '\xc3\xb0eah', '\xc3\xb0eaw', '\xc3\xb0ec', '\xc3\xb0eoda', '\xc3\xb0eode', '\xc3\xb0eoden', '\xc3\xb0eodne', '\xc3\xb0eonydum', '\xc3\xb0on', '\xc3\xb0one', '\xc3\xb0onne', '\xc3\xb0ry', '\xc3\xb0u', '\xc3\xb0uhte', '\xc3\xb0urhgleded', '\xc3\xb0y', '\xc3\xb0\xc3\xa6r', '\xc3\xb0\xc3\xa6re', '\xc3\xb0\xc3\xa6s', '\xc3\xbea', '\xc3\xbeafigan', '\xc3\xbeam', '\xc3\xbean', '\xc3\xbeanc', '\xc3\xbeancia\xc3\xb0', '\xc3\xbeancode', '\xc3\xbeara', '\xc3\xbeas', '\xc3\xbee', '\xc3\xbeeah', '\xc3\xbeeaw', '\xc3\xbeec', '\xc3\xbeegn', '\xc3\xbeegnas', '\xc3\xbeegnum', '\xc3\xbeeh', '\xc3\xbeenden', '\xc3\xbeeode', '\xc3\xbeeoden', '\xc3\xbeeodne', '\xc3\xbeeodnes', '\xc3\xbeeostro', '\xc3\xbeeowned', '\xc3\xbeider', '\xc3\xbein', '\xc3\xbeine', '\xc3\xbeinga', '\xc3\xbeinne', '\xc3\xbeis', '\xc3\xbeisse', '\xc3\xbeolia\xc3\xb0', '\xc3\xbeone', '\xc3\xbeonne', '\xc3\xberage', '\xc3\xberah', '\xc3\xbereanied', '\xc3\xbereaum', '\xc3\xberidda', '\xc3\xberidde', '\xc3\xberistum', '\xc3\xberowigean', '\xc3\xberowode', '\xc3\xbery', '\xc3\xberym', '\xc3\xberymmes', '\xc3\xbeu', '\xc3\xbeuhte', '\xc3\xbeurfe', '\xc3\xbeurh', '\xc3\xbeurhwodon', '\xc3\xbey', '\xc3\xbeyder', '\xc3\xbeysne', '\xc3\xbe\xc3\xa6r', '\xc3\xbe\xc3\xa6re', '\xc3\xbe\xc3\xa6s', '\xc3\xbe\xc3\xa6t', '\xc3\xbe\xc3\xa6tte'], ['A1.3_Dan_T00030_1', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.014705882352941176, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0063025210084033615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0084033613445378148, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.01050420168067227, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0042016806722689074, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012605042016806723, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0063025210084033615, 0.0, 0.0, 0.0063025210084033615, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.029411764705882353, 0.0021008403361344537, 0.0042016806722689074, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0063025210084033615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.01050420168067227, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0063025210084033615, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.01680672268907563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0042016806722689074, 0.0021008403361344537, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.014705882352941176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0063025210084033615, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0063025210084033615, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.01050420168067227, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0042016806722689074, 0.0, 0.0021008403361344537, 0.0063025210084033615, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0063025210084033615, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01050420168067227, 0.0042016806722689074, 0.0021008403361344537, 0.0021008403361344537, 0.0084033613445378148, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01050420168067227, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0063025210084033615, 0.0, 0.0, 0.0, 0.0084033613445378148, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0063025210084033615, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0063025210084033615, 0.0, 0.0, 0.0, 0.014705882352941176, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0063025210084033615, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.042016806722689079, 0.0, 0.012605042016806723, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.012605042016806723, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0, 0.0, 0.0042016806722689074, 0.0021008403361344537, 0.0, 0.0021008403361344537, 0.0, 0.0, 0.0063025210084033615, 0.0, 0.0, 0.0, 0.0, 0.0084033613445378148, 0.0042016806722689074, 0.0042016806722689074, 0.023109243697478993, 0.0063025210084033615], ['A1.3_Dan_T00030_2', 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.017777777777777778, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0044444444444444444, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0077777777777777776, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0055555555555555558, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0033333333333333335, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.012222222222222223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0011111111111111111, 0.022222222222222223, 0.0011111111111111111, 0.0, 0.0, 0.017777777777777778, 0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.012222222222222223, 0.0011111111111111111, 0.0033333333333333335, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0033333333333333335, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0055555555555555558, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.016666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0033333333333333335, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0033333333333333335, 0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0033333333333333335, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.022222222222222223, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.01, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.01, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0077777777777777776, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0077777777777777776, 0.012222222222222223, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0077777777777777776, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.017777777777777778, 0.0011111111111111111, 0.016666666666666666, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015555555555555555, 0.0044444444444444444, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0077777777777777776, 0.0022222222222222222, 0.0066666666666666671, 0.018888888888888889, 0.0], ['A1.3_Dan_T00030_3', 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.06222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0088888888888888889, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0055555555555555558, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0055555555555555558, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0044444444444444444, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.012222222222222223, 0.0, 0.0, 0.0, 0.01, 0.0011111111111111111, 0.0055555555555555558, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.01, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0033333333333333335, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0055555555555555558, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.026666666666666668, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.017777777777777778, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.012222222222222223, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0055555555555555558, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0, 0.0044444444444444444, 0.0088888888888888889, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0088888888888888889, 0.0044444444444444444, 0.0, 0.0077777777777777776, 0.0011111111111111111, 0.0, 0.0066666666666666671, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.017777777777777778, 0.0, 0.0088888888888888889, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0066666666666666671, 0.0, 0.0, 0.012222222222222223, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0011111111111111111, 0.01, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0055555555555555558, 0.0, 0.0011111111111111111, 0.017777777777777778, 0.0022222222222222222], ['A1.3_Dan_T00030_4', 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0077777777777777776, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0033333333333333335, 0.026666666666666668, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0033333333333333335, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0033333333333333335, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0066666666666666671, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.022222222222222223, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0055555555555555558, 0.0, 0.0, 0.0044444444444444444, 0.0011111111111111111, 0.0, 0.0, 0.017777777777777778, 0.0022222222222222222, 0.012222222222222223, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0, 0.0, 0.0055555555555555558, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0055555555555555558, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0055555555555555558, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.02, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.015555555555555555, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0044444444444444444, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0022222222222222222, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.012222222222222223, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0077777777777777776, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0066666666666666671, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0055555555555555558, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0077777777777777776, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0088888888888888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0044444444444444444, 0.0, 0.0011111111111111111, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0055555555555555558, 0.0033333333333333335, 0.0, 0.0022222222222222222, 0.0, 0.0, 0.0022222222222222222, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.014444444444444444, 0.0, 0.0088888888888888889, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0088888888888888889, 0.0011111111111111111, 0.0011111111111111111, 0.0033333333333333335, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022222222222222222, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0033333333333333335, 0.0088888888888888889, 0.0011111111111111111, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011111111111111111, 0.0, 0.0, 0.0077777777777777776, 0.0022222222222222222, 0.0, 0.0088888888888888889, 0.0011111111111111111, 0.0022222222222222222, 0.0011111111111111111, 0.0, 0.0, 0.0, 0.0077777777777777776, 0.026666666666666668, 0.0011111111111111111], ['A1.3_Dan_T00030_5', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0034403669724770644, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0068807339449541288, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.016055045871559634, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0057339449541284407, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0034403669724770644, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0034403669724770644, 0.0, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0034403669724770644, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0022935779816513763, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0034403669724770644, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0, 0.0011467889908256881, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0045871559633027525, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0, 0.008027522935779817, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0034403669724770644, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0034403669724770644, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0011467889908256881, 0.010321100917431193, 0.0011467889908256881, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0068807339449541288, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0068807339449541288, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01834862385321101, 0.0022935779816513763, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0045871559633027525, 0.0011467889908256881, 0.0, 0.0, 0.013761467889908258, 0.0, 0.008027522935779817, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0022935779816513763, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0045871559633027525, 0.0, 0.027522935779816515, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0045871559633027525, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0022935779816513763, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0034403669724770644, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0034403669724770644, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0091743119266055051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0034403669724770644, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0, 0.0057339449541284407, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.014908256880733946, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0091743119266055051, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0068807339449541288, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0022935779816513763, 0.0011467889908256881, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057339449541284407, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0034403669724770644, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0057339449541284407, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.008027522935779817, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0091743119266055051, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01261467889908257, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0034403669724770644, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0057339449541284407, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0045871559633027525, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0034403669724770644, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0045871559633027525, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0022935779816513763, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0034403669724770644, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0091743119266055051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0022935779816513763, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0057339449541284407, 0.0011467889908256881, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.0011467889908256881, 0.014908256880733946, 0.0022935779816513763, 0.0011467889908256881, 0.0057339449541284407, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.017201834862385322, 0.0, 0.0068807339449541288, 0.0, 0.0, 0.0, 0.0, 0.0022935779816513763, 0.0, 0.0091743119266055051, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0011467889908256881, 0.0, 0.0, 0.0, 0.0034403669724770644, 0.0, 0.0, 0.0045871559633027525, 0.0, 0.0, 0.0, 0.0, 0.008027522935779817, 0.0022935779816513763, 0.0, 0.022935779816513763, 0.0011467889908256881]]
    for i in range(1, len(a)):
        for j in range(1, len(a[i])):
            a[i][j] *= 900
    wordlists = matrixtodict(a)
    print 'ge', wordlists[0]['ge'], 'wisdom', wordlists[0]['wisdom'], 'swefnes', wordlists[0]['swefnes']
    print

    print sort(testall(wordlists))
    print
    lalala = groupdivision(wordlists, [[0], [1, 2, 3, 4]])
    temp = testgroup(lalala)
    for key in temp.keys():
        print key, temp[key][:10]

    print
    lalala = groupdivision(wordlists, [[2], [1, 0, 3, 4]])
    temp = testgroup(lalala)
    for key in temp.keys():
        print key, temp[key][:10]


