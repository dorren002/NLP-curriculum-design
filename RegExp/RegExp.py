import re

def searchln(pattern, string):
    global index
    res = re.search(pattern, string)
    if res != None:
        temp = list(res.span())
        if len(index) != 0:
            temp[0] += index[-1][0]+4
            temp[1] += index[-1][0]+4
            index.append(temp)
        else:
            index.append(temp)
        newstr = string[temp[0]+2:temp[1]-2]
        newstr += string[temp[1]:]
        searchln(pattern, newstr)


def searchnest(pattern, string):
    global index
    temp =[]
    nest = re.compile(r'\(.*?\)')
    hasnest = re.search(nest, string)
    if hasnest == None:
        searchln(pattern, string)
    else:
        tempspan = list(hasnest.span())
        str1 = string[:tempspan[0]]+string[tempspan[1]:]
        str2 = string[tempspan[0]:tempspan[1]]
        searchln(pattern, str1)
        for j in range(len(index)):
            index[j][1] += (tempspan[1]-tempspan[0])
        temp = index
        index = []
        searchln(pattern, str2)
        for i in range(len(index)):
            index[i][0] += tempspan[0]
            index[i][1] += tempspan[0]
        for k in range(len(temp)):
            index.append(temp[k])


def println(span, string, lnnum):
    global result
    temp = []
    temp.append(str(lnnum))
    if span[0] > 2:
        temp.append(string[span[0] - 3:span[0]])
    elif span[0] == 0:
        temp.append('')
    else:
        temp.append(string[0:span[0]])

    temp.append('*'+string[span[0]:span[0]+2]+'*')
    temp.append(string[span[0]+2:span[1]-2])
    temp.append('&'+string[span[1]-2:span[1]]+'&')

    if span[1]+2 <= len(string):
        temp.append(string[span[1]:span[1] + 3])
    else:
        temp.append(string[span[1]:])
    result.append(temp)


def saveln(lnnum, span, string):
    if len(span) == 1:
        println(span[0],string, lnnum)
    else:
        for i in range(len(span)):
            println(span[i], string, lnnum)


global result
global index
result = []
pattern = re.compile(r'因为.*?所以')

with open('corpus.txt') as f:
    for i in range(12):
        index = []
        strln = f.readline().strip()
        searchnest(pattern, strln)
        if len(index) > 0:
            saveln(i + 1, index, strln)

s=chr(12288)
max_item = max(len(row[3]) for row in result)
tplt="{0:"+s+"<2}\t{1:"+s+"<3}\t{2:<4}\t{3:"+s+"<"+str(max_item)+"}\t{4:<4}\t{5:<3}"
with open('output1.txt', 'w', encoding='utf-8') as f:
    for i in range(len(result)):
        r = result[i]
        f.writelines(tplt.format(r[0], r[1], r[2], r[3], r[4], r[5])+'\n')
        print(tplt.format(r[0], r[1], r[2], r[3], r[4], r[5]))
