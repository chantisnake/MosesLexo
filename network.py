from collections import defaultdict
def makenetwork(DataList):
    NetWorkData = {}
    for k in range(1, len(DataList[0])):
        tempData = defaultdict(list)
        for i in range(1, len(DataList)):
            for j in range(i+1, len(DataList)):
                if DataList[i][k] == DataList[j][k] or \
                                        DataList[i][k]+'?' == DataList[j][k] or DataList[i][k] == DataList[j][k]+'?':

                    try:
                        tempData[DataList[i][0]].append(DataList[j][0])
                    except KeyError:
                        tempData.update({DataList[i][0]: [DataList[j][0]]})
                    try:
                        tempData[DataList[j][0]].append(DataList[i][0])
                    except KeyError:
                        tempData.update({DataList[j][0]: DataList[i][0]})

        NetWorkData.update({DataList[0][k]: tempData})
    return NetWorkData

def distance(network, key, node1, node2):
    if node1 in network[key][node2]:
        return 1
    else:
        temp = []
        for node in network[key][node2]:
            temp.append(distance(network, key, node, node2))
        return min(temp)


def readdata(content):
    content = content.split('\n')
    Result = []
    for row in content:
        rowlist = row.split('    ')
        Result.append(rowlist)
    return Result


if __name__ == '__main__':
    f = open('network', 'r')
    content = f.read()
    f.close()
    datalist = readdata(content)
    network = makenetwork(datalist)
    print network['Location']['70']
    print distance(network, 'Location', '70', '71')
