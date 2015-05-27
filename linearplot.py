# -*- coding: utf-8 -*-
from __future__ import division
"""
*PLEASE DO NOT CHANGE THIS PROGRAM OR REUSE THE CODE UNLESS YOU THINK YOU HAVE A GOOD ENOUGH HANDLE OF THE CODE*
*THE FUNCTION IN THE PROGRAM WORKS AS A WHOLE, CHANGES CAN LEADS TO SERIOUS BUGS*
this program is the program to reduce the dots needed to draw in the graph in order to let Lexo has a better handle on
rolling window
"""
import timeit
from extra import loadstastic, WordInformation


def get_r(Data, start=0):
    # pace IS A PROBLEM
    """
    this function construct a line with the start and the end of the data,
    and then check how far the data are from the line (mimic the method to get Coefficient of determination)
    the x coordinate starts from :param start, and move at the pace of 1
    * math under this method may not be solid, since I just made it up.

    :param Data: an array that contain all the point we need to plot(only y coordinate, x is determined by start)
    :param start: the point the x coordinate starts.
    :return: an value r range in (\infty, 0]
                this r means how well the data fits with the line. the larger r is, the better the data fits
    """
    # construct the line
    Len = len(Data)
    a = (Data[0] - Data[-1]) / (1 - Len)
    b = Data[0] - a * start

    # calculate r2
    AveData = sum(Data[1:-1]) / Len - 2
    SStot = 0
    SSres = 0
    for i in range(1, Len - 1):
        SStot += abs(Data[i] - AveData)
        SSres += abs(Data[i] - (a * (start + i) + b))
    try:
        return - SSres / SStot
    except ZeroDivisionError:
        # this means that the dot are all (or nearly all because of rounding on the computer) on a horizontal straight line
        return 0  # the highest possible r.


def reduceplot(Datas, start=0, LeastCoDe=0, forcedistant=300):
    """
    this program takes in the original data to plot (only y coordinate,
                                                        assume x will start at :param start, move at the pace of 1)
    and return the point that we actually need to plot (x and y coordinate)

    * the more the data varies, the LeastCoDe should be larger and the forcedistant should be smaller

    :param Datas: the y coordinate of the data you need to plot, assume x will start at :param start, move at the pace of 1
    :param LeastCoDe: stands for Least Coefficient of Determination
            the least value you would accept for r. range in (-\infty,0]
            (the larger this data is, the more sensitive the program, and make the output larger)
            (0 means the graph is entirely accurate (all the dot eliminated will be on the line))
            (this value would not affect the speed of the program)

    :param start: the point x starts
    :param forcedistant: if least distance between two point in the plot
                            (the larger this is, the slower the program will run, the speed increase exponentially in the worst case)
                            (the smaller this is, there will be more point send to D3 to draw)

    :return: the point that we actually need to plot (x and y coordinate)
    """
    Result = [(start, Datas[0])]
    PreviousDraw = start
    Len = len(Datas)

    for i in range(1, Len):
        r2 = get_r(Datas[PreviousDraw: i + 2], start=PreviousDraw)
        if r2 < LeastCoDe or i - PreviousDraw > forcedistant - 1:  # find a point need to plot
            Result.append((i, Datas[i]))  # plot that
            PreviousDraw = i  # record that as the last point plotted
            i += 1  # make sure that we do not check only 2 point in get_r function (:param Data in get_r should be more than 1)

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
    for i in range(information.TotalWordCount - 100):
        temp = templist[i: i + 101]
        PlotData.append(temp.count('the'))

    print
    print 'original dot number:', len(PlotData)
    print
    start = timeit.default_timer()
    ReducedPlotData = reduceplot(PlotData)
    stop = timeit.default_timer()
    print 'the time my computer take to reduce:', stop - start, 'seconds'
    print 'the number of dot we need to draw now:', len(ReducedPlotData)
    print 'the percent of dot we need to draw now:', len(ReducedPlotData) / len(PlotData)
    print 'preview of the original dot:', PlotData[0: 50]
    print 'preview of the dot:', ReducedPlotData[0: 50]
    print
    print 'amazing!'
