import sys
import json
from collections import Counter
import numpy as np

def train_bigram(corpus_path):
    print('loading corpus'.ljust(20, '.'), end=' ')
    corpus = []
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            corpus.append("<s>")
            sentence = line.strip().split()
            for word in sentence:
                corpus.append(word)
            corpus.append("</s>")
    del sentence
    print("done\n"+"training".ljust(20, '.'), end=' ')

    # 统计所有词频次
    counter = Counter()
    cgram = 0
    for word in corpus:
        if word == "</s>":
            continue
        counter[word] += 1
        cgram += 1
    counter = counter.most_common() 

    #构造 词 —— id 对，用于后期查询
    lec = len(counter)
    word2id = {counter[i][0]: i for i in range(lec)} 
    tempk = counter[-1][0]
    tempv = word2id.get("<s>")
    word2id["<s>"] = lec-1
    word2id[tempk] = tempv

    del counter

    # bigram构建
    bigram = np.zeros((lec, lec), dtype=np.float32)
    for i in range(1, len(corpus)):
        w = corpus[i]
        if w == "</s>" or w == "<s>":
            continue
        wid = word2id[w]
        rid = word2id[corpus[i-1]]
        bigram[[rid], [wid]] += 1 
    del corpus

    print('done')
    return bigram, word2id, cgram


class Node(object):
    def __init__(self, data, prob, endpos):
        self.prev = []
        self.data = data
        self.prob = prob
        self.endpos = endpos


def kn(wi1, wi):
    global LM,cgram
    cwi1 = np.sum(LM[wi1])       #count(wi-1)
    N1 = np.sum(LM[wi1]!=0)      #N1+(wi-1 。)
    lamda =  0.75/cwi1 * N1
    return max((LM[wi1][wi] - 0.75),0)/cwi1 + lamda * np.sum(LM[wi])/cgram


def findMAP(new):
    maxx = -1
    index = 0
    for i in range(len(new)):
        if new[i].prob > maxx:
            index = i
    return index


'''
主切分函数，由一次前向最大匹配，一次反向最大匹配和一次最大概率切分构成
参数意义    LM: bigram语言模型  word2id: 词-id对  sentence: 待切分句
'''
def Seg(word2id, sentence):

    MaxLen = 3
    pre = []
    cur = []
    new = [] 
    new.append(Node("<s>", 1, 0))
    for i in range(len(sentence)):
        if i+MaxLen>len(sentence):
            w = sentence[i:]
        else:
            w = sentence[i:i+MaxLen] 
        pre = cur
        cur = new
        new = []
        while len(w)!=0:
            if word2id.get(w)!=None:
                pmax = -1.0
                index = -1
                for j in range(len(cur)):    
                    if cur[j].endpos == i:
                        p = kn(word2id.get(cur[j].data), word2id[w])
                        if(p>pmax):
                            pmax = p
                            index = j
                tmp = Node(w, pmax, i+len(w))
                tmp.prev.append(cur[index])
                new.append(tmp) 
            # 未登录词
            if len(new)==0 and len(w)==1:
                if OOV.get(w)!=None:
                    OOV[w] = len(OOV)+2
                pmax = 0
                index = 0
                for j in range(len(cur)):    
                    if cur[j].endpos == i:
                        p = kn(0, 0, True)
                        if(p>pmax):
                            pmax = p
                            index = j
                tmp = Node(w, pmax, i+len(w))
                tmp.prev.append(cur[index])
                new.append(tmp)
            w=w[:-1]
        for k in range(len(cur)):
            if i!=cur[k].endpos:
                new.append(cur[k])
    
    x = Node("tmp", 0, 0)
    x.prev=new

    res = ''
    while len(x.prev) != 0:
        index = findMAP(x.prev)
        res = x.prev[index].data+ ' ' + res
        x = x.prev[index]
    return res[4:]


if __name__ == "__main__":
    print("Initializing .......")

    global OOV, LM, cgram
    OOV = {}
    traincorpus_path = "corpus.txt"
    testcorpus_path = "test.txt"
    output_path = "SegOutput_KN.txt"

    LM, word2id, cgram = train_bigram(traincorpus_path)

    print("Segmenting..........", end=' ')

    res = ''
    with open(testcorpus_path, encoding='utf-8') as f:
        for sentence in f:
            segs = sentence.strip("\n").split("。")
            for seg in segs:
                print(seg)
                tmp = Seg(word2id, seg)
                res += tmp
                res += '。'
            res +='\n'
    del LM, word2id
    OOVid = {v: k for k, v in OOV.items()}

    del OOVid
    print("done\n"+"Writing into files..", end=' ')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(res)

    if len(OOV) != 0:
        jsobj = json.dumps(OOV, indent=0, ensure_ascii=False)
        with open('OOV.txt', 'w', encoding='utf-8') as f:
            f.write(jsobj)

    print("done")
    print("Segment results are saved in " + output_path + ".")