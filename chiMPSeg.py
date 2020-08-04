import sys
# import json
import numpy as np
from collections import Counter

'''
语言模型训练
'''
def train_bigram(corpus_path):
    print('loading corpus......', end=' ')
    corpus = []
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            corpus.append("<s>")
            sentence = line.strip().split()
            for word in sentence:
                corpus.append(word)
    del sentence
    print("done\n"+"training".ljust(20, '.'), end=' ')

    # 统计所有词频次
    counter = Counter()
    cgram = 0                #gram总数，用于kn平滑计算
    for word in corpus:
        counter[word] += 1
        cgram += 1
    counter = counter.most_common() 
    countmin = counter[-1][0]

    #构造 词 —— id 对，用于后期查询
    lec = len(counter)
    word2id = {counter[i][0]: i for i in range(lec)} 

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
    countmin = word2id[countmin]

    print('done',bigram.shape)
    return bigram, word2id, cgram, countmin

'''
获得最小概率值近似未登陆词
'''
def getpmin(bigram, y):
    x = -1
    for i in range(len(bigram)):
        if bigram[i][y] == 0:
            x = i
            break
    cwi1 = np.sum(bigram[x])       #count(wi-1)
    N1 = np.sum(bigram[x]!=0)      #N1+(wi-1 。)
    lamda =  0.75/cwi1 * N1
    return max((bigram[x][y] - 0.75),0)/cwi1 + lamda * np.sum(bigram[y])/cgram

'''
词网络节点
'''
class Node(object):
    '''
    prev存放当前词节点的上一个节点，唯一
    data存放可能的词
    prob当前位置概率
    endpos当前词的结束位置
    '''
    def __init__(self, data, prob, endpos):
        self.prev = None
        self.data = data
        self.prob = prob
        self.endpos = endpos

'''
计算某点kn平滑后概率，wi1为wi-1   oov=True时当前gram有未登陆词 使用最小概率计算
'''
def kn(wi1, wi, oov=False):
    global LM,pmin
    if wi1 == None or oov==True:
        return pmin
    else:
        cwi1 = np.sum(LM[wi1])       #count(wi-1)
    N1 = np.sum(LM[wi1]!=0)          #N1+(wi-1 。)
    lamda =  0.75/cwi1 * N1
    return max((LM[wi1][wi] - 0.75),0)/cwi1 + lamda * np.sum(LM[wi])/cgram

'''
拉普拉斯平滑,使用时训练中进行了加一
'''
def laplace(wi1, wi):
    global LM
    return (LM[wi1][wi] / np.sum(LM[:,wi]))

'''
为有多个入点的节点选择最佳左邻词，pre为入点集合，wi是当前节点的分词，startpos是wi的起始位置
end=1时为末尾只要寻找最大入点概率即可，
end=2时为未登陆词处理
'''
def getBAW(pre, wi, startpos, end = 0):
    pmax, index = -1.0, -1
    for j in range(len(pre)):    
        if pre[j].endpos == startpos:
            if end == 1:
                pkn = 1
            if end == 2:
                pkn = pmin
            else:
                pkn = kn(word2id.get(pre[j].data), wi)
            p = pre[j].prob * pkn
            if(p>pmax):
                pmax = p
                index = j
    return pmax, index

'''
主切分函数
'''
def Seg(word2id, sentence, MaxLen = 5):
    if sentence == '':
        return ''
    pre = []    # 上一轮自由节点集合，即本轮可以添加节点的节点集合
    cur = []    # 本轮自由节点集合
    cur.append(Node("<s>", 1, 0))
    for i in range(len(sentence)):
        if i+MaxLen>len(sentence):
            w = sentence[i:]
        else:
            w = sentence[i:i+MaxLen] 
            
        pre = cur
        cur = []
        while len(w)!=0:
            if w=='\n':
                p, index = getBAW(pre,-1,i,1)
                tmp = Node(w, p, i+len(w))
                tmp.prev=pre[index]
                cur.append(tmp)
            if word2id.get(w)!=None:
                p, index = getBAW(pre,word2id[w],i)
                if p!=-1:
                    tmp = Node(w, p, i+len(w))
                    tmp.prev=pre[index]
                    cur.append(tmp) 
            # 未登录词
            if len(cur)==0 and len(w)==1:
                if OOV.get(w)==None:
                    OOV[w] = 1
                else:
                    OOV[w] += 1
                p, index = getBAW(pre,-1,i,2)
                tmp = Node(w, p, i+len(w))
                tmp.prev=pre[index]
                cur.append(tmp)
            w=w[:-1]
        # 将未添加新节点的自由节点加入本轮自由节点集合，以备下一轮使用
        for k in range(len(pre)):
            if i!=pre[k].endpos:
                cur.append(pre[k])
    x=cur[0]
    res = '\n'
    while x.prev!=None:
        res = x.prev.data+ ' ' + res
        x = x.prev
    return res[4:]


def spl(string):
    seg = []
    form = 0
    for i in range(len(string)):
        if i == len(string):
                seg.append(string[form:])
                break
        if string[i] =="。" or string[i]=="；":            
            seg.append(string[form:i+1])
            form = i+1
    return seg


if __name__ == "__main__":
    print("Initializing .......")

    global OOV, LM, cgram, pmin
    OOV = {}
    # traincorpus_path = "corpus_for_ass2train.txt"
    # testcorpus_path = "corpus_for_ass2test.txt"
    traincorpus_path = "corpus.txt"
    testcorpus_path = "corpustest.txt"
    output_path = "SegOutput.txt"

    LM, word2id, cgram, countmin = train_bigram(traincorpus_path)
    pmin = getpmin(LM, countmin)
    print("Segmenting..........", end=' ')
    res = ''
    with open(testcorpus_path, encoding='utf-8') as f:
        for sentence in f:
            #语料长度大于100时分为句段进行
            if len(sentence)>100:
                segs = spl(sentence)
                for seg in segs:
                    seg += '\n'
                    tmp = Seg(word2id, seg)
                    res += tmp[:-1]
                res += '\n'
            else:
                tmp = Seg(word2id, sentence)
                res += tmp
    del LM, word2id
    OOVid = {v: k for k, v in OOV.items()}

    del OOVid
    print("done\n"+"Writing into files..", end=' ')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(res[:-1])

    # if len(OOV) != 0:
    #     jsobj = json.dumps(OOV, indent=0, ensure_ascii=False)
    #     with open('OOV.txt', 'w', encoding='utf-8') as f:
    #         f.write(jsobj)

    print("done")
    print("Segment results are saved in " + output_path + ".")