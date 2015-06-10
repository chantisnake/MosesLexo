from scipy.stats.mstats import kruskalwallis
from scipy.stats.stats import kruskal
import numpy.ma as ma


def KWTopWord(Matrixs, Words):
    Len = max(len(matrix) for matrix in Matrixs)
    word_pvalue_dict = {}

    for i in range(1, len(Matrixs[0][0])):
        print Words[i-1]
        samples = []
        for k in range(len(Matrixs)):
            print k
            sample = []
            for j in range(len(Matrixs[k])):
                sample.append(Matrixs[k][j][i])
            print sample
            print
            samples.append(ma.masked_array(sample + [0]*(Len - len(sample)),
                                           mask=[0]*len(sample)+[1]*(Len-len(sample))))
        print samples
        pvalue = kruskalwallis(samples)[1]
        print pvalue
        word_pvalue_dict.update({Words[i-1]: pvalue})
    print word_pvalue_dict



if __name__ == "__main__":
    KWTopWord([[['1', 2, 3, 40], ['2', 3, 2, 20], ['1', 2, 3, 40]], [['1', 2, 3, 4], ['2', 3, 2, 1]]], ['lalala', 'papapa', 'hehehe'])
