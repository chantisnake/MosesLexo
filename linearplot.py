from __future__ import division
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
        SStot += (Data[i] - AveData) ** 2
        SSres += (Data[i] - (a * (start+i) + b)) ** 2

    return 1 - SSres / SStot


def reduceplot(Datas, LeastCoDe, pace=1, start=0, forcedistant=100):
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
    y = range(1000000)
    print len(reduceplot(y, 0.9, forcedistant=50))
