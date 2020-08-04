# 基于ginsim.Word2Vec的模型

import numpy as np
import multiprocessing, time
from gensim.models.word2vec import Word2Vec, LineSentence


##  第一组数据的训练  创建模型并保存
def vcmodel(corpus_file, Vec_dim, model_file, skip_window=2):
    start=time.time()
    model=Word2Vec(LineSentence(corpus_file), sg=1, size=Vec_dim, window=skip_window, negative=5, workers=multiprocessing.cpu_count())
    model.save(model_file)
    end=time.time()

    cha = end-start
    if cha/60>=1:
        chamin = int(cha/60)
        chas = int(cha-chamin*60)
        print("\tdone in 00:{:0>2d}:{}".format(chamin, chas))
    else:
        print("\tdone in {} seconds".format(cha))
    print("\tvocabsize:", model.vocabulary.cum_table.shape[0])
    return model

##  后面九组数据的训练   每次保存
def vamodel(model_file, corpus_file,model):
    tmp = model.vocabulary.cum_table.shape[0]
    newcorpus = LineSentence(corpus_file)
    start = time.time()
    model.build_vocab(newcorpus, update=True)
    model.train(newcorpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save(model_file)
    end = time.time()
    cha = end-start
    if cha / 60 >= 1:
        chamin = int(cha / 60)
        chas = int(cha - chamin * 60)
        print("\tdone in 00:{:0>2d}:{:0>2d}".format(chamin, chas))
    else:
        print("\tdone in {} seconds".format(cha))
    print("\tvocabsize:" ,model.vocabulary.cum_table.shape[0])
    print("\tnewly appended:", model.vocabulary.cum_table.shape[0]-tmp)
    return model


##  查表获得词向量
def getvec(model, word, size):
    try:
        vec = model.wv.__getitem__(word)
    except Exception:
        if u'\u4e00' <= word <= u'\u9fff':
            vec = None
        else:
            vec = getvec(model,word.lower(),size)
    return vec


##  计算相似度
def Similarity(vec1,vec2):
    A = np.sum(vec1*vec2)
    B = np.sqrt(np.sum(vec1**2)+np.sum(vec2**2))
    return A/B


##  将输出写入文件
def writeOutput(cos, file):
    out = ''
    for i in range(len(cos)):
        out += str(cos[i])+'\n'
    with open(file, 'w', encoding='utf-8')as f:
        f.write(out)


##  评估函数 打开测试文件并逐行计算相似度
def Evaluate(model, in_path, out_path, dim):
    res=[]
    with open(in_path, 'r', encoding="utf-8")as f:
        for line in f:
            words = line.split('\t')
            vec1 = getvec(model, words[0], dim)
            vec2 = getvec(model, words[1][:-1], dim)
            if vec1 is None or vec2 is None:
                outline = "{}\t{}\t{}".format(words[0], words[1][:-1], "OOV")
                res.append(outline)
            else:
                vec2 = vec2.reshape(1, 100)
                cosin=model.wv.cosine_similarities(vec1,vec2)[0]
                outline = "{}\t{}\t{}".format(words[0], words[1][:-1], cosin)
                res.append(outline)
    writeOutput(res, out_path)


if __name__ == "__main__":
    corpus_path = "wiki/wiki_"
    model_file = "model/model_1"
    test_file = "pku_sim_test.txt"
    out_path = "out/ouput__"
    window = 2
    dim=100
    epoch=10
    out_file = "out/output__9.txt"
    m = Word2Vec.load("model/model_1")
    Evaluate(m, test_file, out_file, dim)
    start=time.time()
    for i in range(epoch):
        print("step" + str(i) + ":", end=' ')
        corpus_file = corpus_path + str(i) +'.txt'
        out_file = out_path + str(i) + ".txt"
        if i==0:
            model = vcmodel(corpus_file, dim, model_file,window)
            Evaluate(model, test_file, out_file, dim)
        else:
            model=vamodel(model_file,corpus_file,model)
            Evaluate(model, test_file, out_file, dim)
        print("sleep:  20s")
        time.sleep(20)
    
    end=time.time()
    cha=end-start
    if cha / 60 >= 1:
        chamin = int(cha / 60)
        chas = int(cha - chamin * 60)
        if chamin/60 >=1:
            chah = int(chamin/60)
            chamin = int(chamin - chah * 60)
        else:
            chah = 0
        print("Totally used {:0>2d}:{:0>2d}:{:0>2d}".format(chah, chamin, chas))
    else:
         print("Totally used {} seconds".format(cha))
