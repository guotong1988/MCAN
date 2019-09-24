import pickle
from collections import defaultdict
import logging
import gensim
import numpy as np
from random import shuffle
from gensim.models.word2vec import Word2Vec
logger = logging.getLogger('relevance_logger')


def build_multiturn_data(trainfile, word2id = None, isshuffle=False):
    all_data = []
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

            data = {"y" : label, "c": message, "r": response}
            all_data.append(data)
            total += 1
            if total % 10000 == 0:
                print(total)

    logger.info("processed dataset with %d question-answer pairs " %(len(all_data)))
    logger.info("vocab size: %d" % (len(word2id)))
    if isshuffle == True:
        shuffle(all_data)
    return all_data,


import json

def ParseMultiTurn():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    f = open("C:\\Users\\gt\\Desktop\\Dialogue-master\\Research\\vocab-cn.txt", mode="r", encoding="utf-8")
    word2id = {}
    lines = f.readlines()
    for i,line in enumerate(lines):
        word2id[line.rstrip()] = i
    word2id["_EOS_"] = len(word2id)
    print("_EOS_",word2id["_EOS_"])
    data1 = build_multiturn_data("train.txt", word2id=word2id, isshuffle=False)
    data2 = build_multiturn_data("dev.txt", word2id=word2id, isshuffle=False)
    data3 = build_multiturn_data("test.txt", word2id=word2id, isshuffle=False)

    pickle.dump([data1, data2, data3], open("douban_data.pkl",'wb'))

def make_emd():
    f = open("C:\\Users\\gt\\Desktop\\Dialogue-master\\Research\\vocab-cn.txt", mode="r", encoding="utf-8")
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
    ParseMultiTurn()
    make_emd()