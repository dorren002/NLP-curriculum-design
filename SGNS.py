# https://www.leiphone.com/news/201706/PamWKpfRFEI42McI.html

import torch,math,random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.autograd import Variable

# 建立词表
def buildVocab(corpus, drop=False, th=0.001):
    counter = Counter()
    for article in corpus:
        words = article.split(' ')
        for word in words:
            counter[word] += 1
    del article, words
    counter = counter.most_common()

    size = len(counter)
    vocab = {counter[i][0]: i for i in range(size)}
    unigram = {word:count for word,count in counter}  # unigram 模型
    # 高频词抽样概率
    if drop:
        pdrop = {word:1-math.sqrt(th/count) for word,count in counter}
    else:
        pdrop = None
    return vocab,unigram,pdrop

# 损失函数
class NSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs_, positive_, negative_,nNS=5):
        batch_size,embed_size=inputs_.shape
        input_vectors=inputs_.view(batch_size,embed_size, 1)  # reshape为可以相乘的形状
        positive_vectors=positive_.view(batch_size,1,embed_size)
        out_loss=(torch.bmm(positive_vectors,input_vectors)).sigmoid().log()
        out_loss=out_loss.squeeze()
        negative_loss=(torch.bmm(negative_.neg(),input_vectors)).sigmoid().log()
        negative_loss=negative_loss.squeeze().sum(-1)
        output_loss=out_loss+negative_loss
        output_loss=Variable(output_loss,requires_grad=True)
        return output_loss

# skipgram主类
class SkipGramNS(nn.Module):
    def __init__(self, n_vocab, n_vector, noise_dist=None):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_vecotr = n_vector
        self.noise_dist = noise_dist

        self.input_embed = nn.Embedding(n_vocab, n_vector)
        self.output_embed = nn.Embedding(n_vocab,n_vector)

        self.input_embed.weight.data.uniform_(-1,1)
        self.output_embed.weight.data.uniform_(-1,1)

    # 获得输入的向量表示
    def forward_input(self, input_):
        input_ = Variable(input_)
        input_embedded = self.input_embed(input_)
        return input_embedded

    # 获得正例的向量表示
    def forward_output(self, output_):
        output_ = Variable(output_)
        output_embedded = self.output_embed(output_)
        return output_embedded

    # 随机获得负例的向量表示
    def forward_negative(self, batch_size, n_samples):
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
        negative_samples = torch.multinomial(noise_dist, batch_size*n_samples,replacement=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        negative_samples = negative_samples.to(device)
        negative_vectors = self.forward_output(negative_samples).view(batch_size,n_samples,self.n_vecotr)
        return  negative_vectors

# 高频词抽样
def Subsampling(target_words):
    global pdrop
    if pdrop==None:
        return target_words
    else:
        train_words = []
        for word in target_words:
            if random.random() < pdrop[word]:
                train_words.append(word)
        return train_words

# 构造（input_word, output_word）形式的迭代器
def get_batches(corpus, skip_window=3):
    for idx in range(0,len(corpus)):
        ii=0
        x=[]
        left = idx-skip_window if idx-skip_window>=0 else 0
        right = idx+skip_window if idx+skip_window<=len(corpus) else len(corpus)
        batch = corpus[left:right]
        batch_x = batch[ii]
        batch_y = batch[:ii] + batch[ii+1:]
        x.extend([batch_x]*len(batch_y))
        yield x, batch_y

# 训练函数
def train(model, criterion, n_NS=5):
    global vocab
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = 0
    epochs = 5
    for e in range(epochs):
        for article in corpus:
            article = article.split(' ')
            for input, targets in get_batches(article):
                steps+=1
                batch_size=len(input)
                input = torch.LongTensor([vocab[i] for i in input])
                targets=torch.LongTensor([vocab[target] for target in targets])
                input, targets = input.to(device), targets.to(device)

                input = model.forward_input(input)
                targets = model.forward_output(targets)
                negative_samples = model.forward_negative(batch_size, n_NS)

                loss = criterion(Variable(input), Variable(targets), Variable(negative_samples))
                loss = Variable(loss)
                optimizer.zero_grad() # 清空上一步的残余更新参数值
                loss.detach()       # 误差反向传播, 计算参数更新值
                optimizer.step()      # 将参数更新值施加到 net 的 parameters 上
    torch.save(model, "model")

# 输出写入文件
def writeOutput(cos, file):
    out = ''
    for i in range(len(cos)):
        out += str(cos[i])+'\n'
    with open(file, 'w', encoding='utf-8')as f:
        f.write(out)

# 相似度计算
def Similarity(vec1,vec2):
    c=vec1.T
    A = vec1.T*vec2
    B = torch.sqrt((vec1**2).sum()+(vec2**2).sum())
    return A/B

# 获得输入词的向量表示
def getvec(model, word, size):
    global vocab
    if vocab.get(word)!=None:
        inp = torch.LongTensor(vocab[word]).cuda()
        vec = model.forward_input(inp)
    else:
        return None
    return vec

# 评估函数
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
                cosin=Similarity(vec1,vec2)
                outline = "{}\t{}\t{}".format(words[0], words[1][:-1], cosin)
                res.append(outline)
    writeOutput(res, out_path)


if __name__ == "__main__":
    global vocab, unigram, pdrop, n_total
    corpus_ = "test.txt"
    test_ = "pku_sim_test.txt"
    out_ = "out1.txt"
    with open(corpus_, "r", encoding="utf-8")as f:
        corpus = f.readlines()
    vocab, unigram, pdrop = buildVocab(corpus, drop=True)
    n_vocab = len(vocab)
    n_total = sum(unigram.values())
    n_vector = 100

    noise_dist = np.array(sorted(unigram.values(),reverse=True))
    noise_dist = torch.from_numpy(noise_dist**0.75/n_total**0.75)
    model = SkipGramNS(n_vocab, n_vector, noise_dist)
    model = model.to("cuda")
    criterion = NSLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    train(model,criterion)
    Evaluate(model,test_,out_,n_vector)