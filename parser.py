# 语料预处理

from gensim.corpora import WikiCorpus
import jieba, jieba.analyse, re
from collections import Counter

# xml转txt
def xml2txt(f_name, out_name):
    output = open(out_name, 'w', encoding='utf-8')
    wiki = WikiCorpus(f_name, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        str_line = ' '.join(text)
        output.write(str_line+'\n')

# 繁转简
# 命令行   opencc -i zhwiki-articles.txt -o wiki.txt -c zht2zhs.ini

# 分词
def Seg(f_name, out_name):
    with open(f_name, 'r', encoding="utf8")as f1:
        with open(out_name, 'w', encoding="utf8")as f2:
            for line in f1:
                line_seg = " ".join(jieba.cut(line))
                f2.writelines(line_seg)

# 去英文
def endel(in_file, out_file):
    with open(in_file, 'r', encoding="utf8")as f1:
        with open(out_file, 'w', encoding="utf-8")as f2:
            for line in f1:
                out = re.sub("[A-Za-z0-9]","",line)
                out2= re.sub(" +"," ",out)
                f2.writelines(out2)


if __name__=="__main__":
    xml_file = 'zhwiki-articles.xml.bz2'
    zht_file = 'zhwiki-articles.txt'
    xml2txt(xml_file, zht_file)

    zhs_file = 'zhwiki-simple.txt'
    zhss_file = 'zhwiki-simple-seg.txt'
    zhss_noen = 'corpus.txt'
    Seg(zhs_file, zhss_file)
    endel(zhss_file, zhss_noen)
    with open(zhss_file, 'r', encoding="utf-8")as f:
        corpus=f.readlines()
    # 将数据分为10个部分
    for i in range(10):
        with open("wiki/wiki_"+str(i)+".txt", 'w', encoding="utf-8")as f2:
            for j in range(36000):
                f2.write(corpus[i*36000+j])



