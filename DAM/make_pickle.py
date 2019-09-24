import pickle
from collections import defaultdict
import logging
import gensim
import numpy as np
from random import shuffle
from gensim.models.word2vec import Word2Vec
logger = logging.getLogger('relevance_logger')


def sub_makedata_douban(trainfile, word2id = None, isshuffle=False):
    all_data_y = []
    all_data_c = []
    all_data_r = []
    total = 1
    with open(trainfile, mode = 'r', encoding = 'utf-8') as f:
        for line in f:
            line = line.replace("_","")
            parts = line.strip().split("\t")

            label = parts[0]
            message = []
            for i in range(1,len(parts)-1,1):
                for char in parts[i]:
                    if char!=" ":
                        if char in word2id:
                            message.append(word2id[char])
                        else:
                            message.append(word2id["[UNK]"])
                message.append(word2id["_EOS_"])
            response = []
            for char in parts[-1]:
                if char!=" ":
                    if char in word2id:
                        response.append(word2id[char])
                    else:
                        response.append(word2id["[UNK]"])

            all_data_c.append(message)
            all_data_r.append(response)
            all_data_y.append(label)
            total += 1
            if total % 10000 == 0:
                print(total)
    all_data = {"y": all_data_y, "c": all_data_c, "r": all_data_r}
    logger.info("processed dataset with %d question-answer pairs " %(len(all_data)))
    logger.info("vocab size: %d" % (len(word2id)))
    if isshuffle == True:
        shuffle(all_data)
    return all_data



def makedata_douban():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    f = open("../sample_data/vocab-cn.txt", mode="r", encoding="utf-8")
    word2id = {}
    lines = f.readlines()
    for i,line in enumerate(lines):
        word2id[line.rstrip()] = i
    word2id["_EOS_"] = len(word2id)
    print("_EOS_",word2id["_EOS_"])
    data1 = sub_makedata_douban("train.txt", word2id=word2id, isshuffle=False)
    data2 = sub_makedata_douban("dev.txt", word2id=word2id, isshuffle=False)
    data3 = sub_makedata_douban("test.txt", word2id=word2id, isshuffle=False)

    pickle.dump([data1, data2, data3], open("douban_data.pkl",'wb'))

def makedata_quora():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    f = open("../sample_data/vocab-cn.txt", mode="r", encoding="utf-8")
    word2id = {}
    lines = f.readlines()
    for i,line in enumerate(lines):
        word2id[line.rstrip()] = i
    word2id["_EOS_"] = len(word2id)
    print("_EOS_",word2id["_EOS_"])
    data1 = sub_makedata_douban("train.txt", word2id=word2id, isshuffle=False)
    data2 = sub_makedata_douban("dev.txt", word2id=word2id, isshuffle=False)
    data3 = sub_makedata_douban("test.txt", word2id=word2id, isshuffle=False)

    pickle.dump([data1, data2, data3], open("douban_data.pkl",'wb'))

# 从词向量搞出字向量，BERT字典的
def make_emd():
    f = open("../sample_data/vocab-cn.txt", mode="r", encoding="utf-8")
    word_list = []
    lines = f.readlines()
    for i, line in enumerate(lines):
        word_list.append(line.rstrip())
    f = open("C:\\Users\\gt\\Desktop\\Dialogue-master\\DAM\\data\\cn_wordvec.txt",mode="r",encoding="utf-8")

    all_char2vec = {}

    def to_float(input):
        return float(input)

    for i,line in enumerate(f):
        if i == 0:
            continue
        if i%100000==0:
            print(i)
        splits = line.rstrip().split(" ")
        if len(splits[0])==1:
            all_char2vec[splits[0]] = list(map(to_float, splits[1:]))

    wordvec_list = []

    for word in word_list:
        if word in all_char2vec:
            wordvec_list.append(all_char2vec[word])
        else:
            wordvec_list.append([0]*300)

    pickle.dump(wordvec_list, open("char_embedding.pkl", 'wb'))

    logger.info("dataset created!")

if __name__=="__main__":
    makedata_douban()
    make_emd()