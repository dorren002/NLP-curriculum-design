import sys
import json
import train
import numpy as np 

'''
反向匹配指定字符串
参数意义   self: 原始字符串   old: 替换目标   new: 替换词   *max: 替换次数 可选(防止"北京市北京大学"内"北京"全部被替换)  
'''
def rreplace(self, old, new, *max):
    count = len(self)
    if max and str(max[0]).isdigit():
        count = max[0]
    return new.join(self.rsplit(old, count))

'''
前向最大匹配算法
参数意义    s1: 待分词句子    word2id: 词id对   MaxLen: 最大长度
'''
def FMM(s1, word2id, MaxLen=3):
    global OOV
    s2 = np.array([])
    while len(s1) != 0:
        if len(s1) <= MaxLen:
            w = s1
        else:
            w = s1[:MaxLen]

        while len(w) != 0:
            if word2id.get(w) != None:
                s2 = np.append(s2, word2id[w])
                s1 = s1.replace(w, '', 1)
                break
            #处理未登录词
            elif len(w) == 1:
                if OOV.get(w) == None:
                    tmp = len(OOV) + 2
                    OOV[w] = tmp
                s2 = np.append(s2, OOV[w]*(-1))
                s1 = s1.replace(w, '', 1)
                break
            else:
                w = w[:-1]

    return s2


'''
反向最大匹配算法
参数意义    s1: 待分词句子    word2id: 词-id对   MaxLen: 最大长度
'''
def BMM(s1, word2id, MaxLen=3):
    global OOV
    s2 = np.array([])
    while len(s1) != 0:
        if len(s1) <= MaxLen:
            w = s1
        else:
            w = s1[-MaxLen:]

        while len(w) != 0: 
            if word2id.get(w) != None:
                s2 =np.insert(s2, 0, word2id[w])
                s1 = rreplace(s1, w, '', 1)
                break
            elif len(w) == 1:
                if OOV.get(w) == None:
                    tmp = len(OOV) + 2
                    OOV[w] = tmp
                s2 = np.insert(s2, 0, OOV[w]*(-1))
                s1 = rreplace(s1, w, '', 1)
                break
            else:
                w = w[1:]

    return s2


def kn(wi1, wi):
    global LM,cgram
    cwi1 = np.sum(LM[wi1])       #count(wi-1)
    N1 = np.sum(LM[wi1]!=0)      #N1+(wi-1 。)
    lamda =  0.75/cwi1 * N1
    return max((LM[wi1][wi] - 0.75),0)/cwi1 + lamda * np.sum(LM[wi])/cgram


'''
最大概率切分
参数意义  LM : bigram模型   MAP: 词网络
'''
def MP(fm, bm):
     ls = max(fm.size, bm.size)
     AP = np.ones((2, ls), dtype=np.float32) * (-1)
     BAW = np.array([])
     for i in range(ls+1):
         if i==0:
             AP[0][0] = 1.0
             continue

         if fm[i-1] < 0:
             AP[0][i] = AP[0][i - 1]
         else:
             p = kn(fm[i-1],fm[i])
             AP[0][i] = AP[0][i-1] * p

         if i-1 >= bm.size:
             continue

         if bm[i-1] == fm[i-1]:
             continue
         else:
             if AP[1][i-1] != -1:
                 if bm[i-1] < 0:
                     AP[1][i] = AP[1][i-1]
                 else:
                     p = kn(fm[i-1],fm[i])
                     AP[1][i] = AP[1][i-1] * p
             else:
                 if bm[i-1] > 0:
                     p = kn(fm[i-1],fm[i])
                     AP[1][i] = AP[0][i-1] * p
     i -= 1
     while i>=0:
         if AP[0][i] > AP[1][i]:
             np.insert(BAW, 0, fm[i-1])
         else:
             np.insert(BAW, 0, bm[i-1])
     return BAW


'''
主切分函数，由一次前向最大匹配，一次反向最大匹配和一次最大概率切分构成
参数意义    LM: bigram语言模型  word2id: 词-id对  sentence: 待切分句
'''
def Seg(word2id, sentence):
    global LM
    if sentence == '':
        return np.array([None])
    fm = FMM(sentence, word2id)
    bm = BMM(sentence, word2id)
    if fm.all() == bm.all():
        return fm
    else:
        if fm.size > bm.size:
            res = MP(fm, bm)
        else:
            res = MP(bm, fm)
        return res


if __name__ == "__main__":
    print("Initializing .......")
    
    global OOV, LM, cgram
    OOV = {}
    traincorpus_path = "corpus.txt"
    testcorpus_path = "corpustest.txt"
    output_path = "SegOutput_KN.txt"
    
    LM, word2id, cgram = train.train_bigram(traincorpus_path)
    # LM = np.loadtxt(open("bigram.csv","rb"),delimiter=",",skiprows=0)
    # word2id = json.load(open("word2id.txt", encoding="utf-8"))

    print("Segmenting..........", end=' ')
    SegResults = np.array([], dtype=np.int16)
    with open(testcorpus_path, encoding='utf-8') as f:
        for sentence in f:
            segs = sentence.strip("\n").split("。")
            for seg in segs:
                SegResult = Seg(word2id, seg)
                if SegResult.any()!=None:
                    SegResults = np.append(SegResults, SegResult)
            SegResults = np.append(SegResults, [-1])
    id2word = {i: w for w, i in word2id.items()}
    del LM, word2id
    OOVid = {v: k for k, v in OOV.items()}

    res = ""
    for id in SegResults:
        if id == -1:
            res += '\n'
            continue
        elif id < -1:
            word = OOVid[(id*(-1))]
        else:
            word = id2word[id]
        res += word
        res += " "

    del id2word, OOVid
    print("done\n"+"Writing into files..", end=' ')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(res)

    if len(OOV) != 0:
        jsobj = json.dumps(OOV, indent=0, ensure_ascii=False)
        with open('OOV.txt', 'w', encoding='utf-8') as f:
            f.write(jsobj)

    print("done")
    print("Segment results are saved in " + output_path + ".")
