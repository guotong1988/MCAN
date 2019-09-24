from Research.tokenization import *
import numpy as np

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # zgwang: starting point of each batch



class DataReader():
    def __init__(self, data_file, vocab_file, batch_size=16):
        self.tokenizer = FullTokenizer(vocab_file,do_lower_case=True)
        self.label_list = []
        self.turns_str_list = []
        self.turns_ids_list = []
        self.response_str_list = []
        self.response_ids_list = []
        self.bert_input_list = []
        f = open(data_file, mode="r", encoding="utf-8")
        for line in f:
            line = line.rstrip()
            splits = line.split("\t")
            label = splits[0]
            turns_str = splits[1:-1]
            response_str = splits[-1]
            response_str = re.sub(" ","",response_str)
            turns_str = [re.sub(" ","",s) for s in turns_str]
            label = int(label)
            turns_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s)) for s in turns_str]
            response_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(response_str))
            bert_input = []
            bert_input.extend(self.tokenizer.convert_tokens_to_ids(["[CLS]"]))
            for ids in turns_ids:
                bert_input.extend(ids)
                bert_input.extend(self.tokenizer.convert_tokens_to_ids(["[SEP]"]))
            bert_input.extend(response_ids)
            bert_input.extend(self.tokenizer.convert_tokens_to_ids(["[SEP]"]))

            self.label_list.append(label)
            self.turns_str_list.append(turns_str)
            self.response_str_list.append(response_str)
            self.response_ids_list.append(response_ids)
            self.turns_ids_list.append(turns_ids)

        self.num_data = len(self.label_list)

        batch_spans = make_batches(self.num_data, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_batch = []
            for i in range(batch_start, batch_end):
                cur_batch.append(self.bert_input_list[i])
            self.batches.append(cur_batch)

        print()

    def get_next_batch(self):
        print()
