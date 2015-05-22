# -*- coding: utf-8 -*-
from __future__ import division
from extra import loadstastic, WordInformation

import numpy


def get_r(Data, pace=1, start=0):
    # pace IS A PROBLEM
    # get the function
    Len = len(Data)
    a = (Data[0] - Data[-1]) / (1 - Len)
    b = Data[0] - a*start

    # calculate r2
    AveData = sum(Data) / Len
    SStot = 0
    SSres = 0
    for i in range(Len):
        SStot += abs(Data[i] - AveData)
        SSres += abs(Data[i] - (a * (start+i) + b))
    try:
        return 1 - SSres / SStot
    except ZeroDivisionError:
        return 1


def reduceplot(Datas, LeastCoDe=0.9, pace=1, start=0, forcedistant=100):
    Result = [(start, Datas[0])]
    PreviousDraw = start
    Len = len(Datas)

    for i in range(Len):
        r2 = get_r(Datas[PreviousDraw: i+2], pace=pace, start=PreviousDraw)
        if r2 < LeastCoDe or i - PreviousDraw > forcedistant - 1:
            Result.append((i, Datas[i]))
            PreviousDraw = i
            i += 1

    Result.append((Len - 1, Datas[Len - 1]))

    return Result


if __name__ == '__main__':
    FileName = 'moby_dick.txt'
    f = open(FileName, 'r')
    content = f.read()
    f.close()
    Data = loadstastic(content)
    information = WordInformation(Data, FileName)
    information.list()
    templist = content.split(' ')
    PlotData = []
    for i in range(information.TotalWordCount-100):
        temp = templist[i: i+101]
        PlotData.append(temp.count('a'))
    print
    print 'PlotData ready'
    print
    print len(reduceplot(PlotData)) / len(PlotData)
