import sys
import os
import time

import pickle
import tensorflow as tf
import numpy as np

import DAM.utils.reader as reader
import DAM.utils.evaluation as eva


def read_word2id():
    f = open("data/ubuntu/word2id",mode="r",encoding="utf-8")
    index = 0
    word2id = {}
    id2word = {}
    for line in f:
        if index%2==0:
            word = line.rstrip()
        if index%2==1:
            word2id[word]=line.rstrip()
            id2word[line.rstrip()]=word
        index+=1
    word2id["_PAD_"] = "0"
    id2word["0"] = "_PAD_"
    return word2id,id2word


def read_word2id_douban():
    f = open("data/douban/word2id",mode="r",encoding="utf-8",errors="ignore")
    word2id = {}
    id2word = {}
    i = 0
    for line in f:
        i+=1
        if i>=77497:
          splits = line.rstrip().split("\t")
          if len(splits)!=2:
              continue
          word2id[splits[0]] = splits[1]
          id2word[splits[1]] = splits[0]
    # word2id["_PAD_"] = "0"
    # id2word["0"] = "_PAD_"
    print("word2id",len(word2id))
    return word2id,id2word

def test(conf, _model):
    
    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print('starting loading data')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))    
    print('finish loading data')

    test_batches = reader.build_batches(test_data, conf)

    print("finish building test batches")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # refine conf
    test_batch_num = len(test_batches["response"])

    print('configurations: %s' %conf)

    word2id,id2word = read_word2id_douban()

    _graph = _model.build_graph()
    print('build graph sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    with tf.Session(graph=_graph) as sess:
        #_model.init.run();
        _model.saver.restore(sess, conf["init_model"])
        print("sucess init %s" %conf["init_model"])

        batch_index = 0
        step = 0

        score_file_path = conf['save_path'] + 'score.test'
        score_file = open(score_file_path, 'w')

        print('starting test')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        for batch_index in range(test_batch_num):
                
            feed = { 
                _model.turns: test_batches["turns"][batch_index],
                _model.tt_turns_len: test_batches["tt_turns_len"][batch_index],
                _model.every_turn_len: test_batches["every_turn_len"][batch_index],
                _model.response: test_batches["response"][batch_index],
                _model.response_len: test_batches["response_len"][batch_index],
                _model.label: test_batches["label"][batch_index]
                }   
                
            scores = sess.run(_model.logits, feed_dict = feed)
                    
            for i in range(conf["batch_size"]):
                context = ""
                for ii in range(test_batches["tt_turns_len"][batch_index][i]):
                    for word_id in test_batches["turns"][batch_index][i][ii]:
                        if word_id!=0:
                            context+=id2word[str(word_id)] + " "
                    context += "_EOS_ "
                response = ""
                for word_id in  test_batches['response'][batch_index][i]:
                    if word_id != 0:
                        response += id2word[str(word_id)] + " "
                score_file.write(
                    str(scores[i]) + '\t' + 
                    str(test_batches["label"][batch_index][i]) + '\t' +
                    context + '\t' +
                    response + '\n')

        score_file.close()
        print('finish test')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        
        #write evaluation result
        result = eva.evaluate(score_file_path)
        result_file_path = conf["save_path"] + "result.test"
        with open(result_file_path, 'w') as out_file:
            for p_at in result:
                out_file.write(str(p_at) + '\n')
        print('finish evaluation')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        

                    
