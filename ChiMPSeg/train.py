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

    # # laplace smoothing(<s>|？不计数)
    # bigram += 1
    # bigram[:, lec-1] -= 1

    # # kn smoothing
    # for i in range(lec):
    #     for j in range(lec):
    #         if j == word2id['<s>']:
    #             continue
    #         sumwi1 = np.sum(lm[i])
    #         N1 = np.sum(lm[i]!=0)
    #         lamda =  0.75/sumwi1 * N1
    #         lm[i][j] = max((lm[i][j] - 0.75),0)/sumwi1 + lamda * np.sum(lm[j])/cgram

    # for i in range(lec):
    #     bigram[:,i] /= bigram[:,i].sum()

    print('done')
    return bigram, word2id, cgram
