import numpy as np
import random
from collections import namedtuple
from data_utility import DataUtility

Data = namedtuple('Data', ['in_data', 'words_num', 'letters_num', 'out_data'])


class DataFeederContext(object):
    def __init__(self, config, vocab_file_in_words="resource/vocab/vocab_in_words",
                 vocab_file_in_letters="resource/vocab/vocab_in_letters",
                 vocab_file_out="resource/vocab/vocab_out",
                 corpus_file_in_words="resource/train_data/train_in_ids_words",
                 corpus_file_in_letters="resource/train_data/train_in_ids_letters",
                 corpus_file_out="resource/train_data/train_out_ids"):
        # Use bucketing to reduce padding
        self.PAD_ID = 0
        self.Buckets = config.buckets
        self.data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words,
                                        vocab_file_in_letters=vocab_file_in_letters,
                                        vocab_file_out=vocab_file_out, max_sentence_length=0)

        corpus_in_words = self.load_corpus(corpus_file_in_words)
        corpus_in_letters = self.load_corpus(corpus_file_in_letters)
        corpus_out = self.load_corpus(corpus_file_out)
        self.all_data = [[] for _ in self.Buckets]  # all_data which is divided into different bukets
        for i in range(len(corpus_in_words)):
            in_words_array = corpus_in_words[i].strip().split()
            in_letters_array = corpus_in_letters[i].strip().split()
            if len(in_letters_array) + len(in_words_array) == 0:
                continue
            if len(in_letters_array) <= self.Buckets[-1]:
                for bucketid, bucketlength in enumerate(self.Buckets):
                    if len(in_letters_array) + len(in_words_array) <= bucketlength:
                        in_data = in_words_array + in_letters_array + [self.PAD_ID] * (bucketlength - len(in_words_array) - len(in_letters_array))
                        words_num = len(in_words_array)
                        letters_num = len(in_letters_array)
                        out_data = corpus_out[i].strip()
                        data = Data(in_data=in_data, words_num=words_num, letters_num=letters_num, out_data=out_data)
                        self.all_data[bucketid].append(data)
                        break
                    if len(in_letters_array) + len(in_words_array) > self.Buckets[-1]:
                        if len(in_letters_array) < self.Buckets[-1]:
                            in_data = in_words_array[-(self.Buckets[-1] - len(in_letters_array)):] + in_letters_array
                        else:
                            in_data = in_letters_array
                        words_num = self.Buckets[-1] - len(in_letters_array)
                        letters_num = len(in_letters_array)
                        out_data = corpus_out[i].strip()
                        data = Data(in_data=in_data, words_num=words_num, letters_num=letters_num, out_data=out_data)
                        self.all_data[self.Buckets.index(self.Buckets[-1])].append(data)
                        break

        self.train_bucket_sizes = [len(self.all_data[b]) for b in range(len(self.Buckets))]
        print ("bucket size = " + str(self.train_bucket_sizes))
        self.num_samples = float(sum(self.train_bucket_sizes))
        self.train_buckets_scale = [sum(self.train_bucket_sizes[:i + 1]) / self.num_samples for i in range(len(self.train_bucket_sizes))]
        print ("bucket_scale = " + str(self.train_buckets_scale))
        print ("samples num = " + str(self.num_samples))
        self.current_batch_index = [0 for i in range(len(self.Buckets))]
        self.tmp_bucket_sizes = [len(self.all_data[b]) for b in range(len(self.Buckets))]
        self.tmp_bucket_scale = [sum(self.train_bucket_sizes[:i + 1]) / self.num_samples for i in range(len(self.train_bucket_sizes))]

    def load_corpus(self, corpus_file_in):
        corpus_array = []
        with open(corpus_file_in, mode="r") as f:
            for line in f:
                corpus_array.append(line)
        return corpus_array

    def init_bucket_param(self):
        self.tmp_bucket_sizes = [len(self.all_data[b]) for b in range(len(self.Buckets))]
        self.tmp_bucket_scale = [sum(self.train_bucket_sizes[:i + 1]) / self.num_samples for i in range(len(self.train_bucket_sizes))]
        self.current_batch_index = [0 for i in range(len(self.Buckets))]
        for i in range(len(self.all_data)):
            random.shuffle(self.all_data[i])

    def maskWeight(self, data):
        in_data = data.in_data
        in_letters_id = in_data[data.words_num : data.words_num + data.letters_num]
        in_letters = [self.data_utility.id2token_in[int(id)] for id in in_letters_id]
        in_word = ''.join(in_letters)
        out_word = self.data_utility.id2token_out[int(data.out_data)]
        return 15.0 if in_word == out_word else 5.0

    def next_batch_fixmask(self, batch_size=32):

        while True:
            tmp_num_samples = float(sum(self.tmp_bucket_sizes))
            self.tmp_bucket_scale = [sum(self.tmp_bucket_sizes[:i + 1]) / tmp_num_samples for i in
                                     range(len(self.train_bucket_sizes))]
            random_number_01 = np.random.random_sample()
            #print random_number_01
            bucket_id = min([i for i in range(len(self.tmp_bucket_scale)) if self.tmp_bucket_scale[i] > random_number_01])
            if self.current_batch_index[bucket_id] + batch_size > self.train_bucket_sizes[bucket_id]:
                self.tmp_bucket_sizes[bucket_id] = 0

            else:
                self.tmp_bucket_sizes[bucket_id] -= batch_size
                break
        # print(self.tmp_bucket_sizes)
        # print(self.tmp_bucket_scale)

        current_batch_length = self.Buckets[bucket_id]
        start_index, end_index = self.current_batch_index[bucket_id], self.current_batch_index[bucket_id] + batch_size
        data_batch = self.all_data[bucket_id][start_index:end_index]
        input_array = np.array([data.in_data for data in data_batch], dtype=np.int32)
        output_array = np.array([[self.PAD_ID] * (data.words_num - 1) +
                                 [data.out_data] * (data.letters_num + 1 if data.words_num > 0 else data.letters_num) +
                                 [self.PAD_ID] * (current_batch_length - data.words_num - data.letters_num)
                                 for data in data_batch], dtype=np.int32)
        mask_array = [[0.0] * (data.words_num - 1) + ([1.0 if data.letters_num > 0 else 10.0] if data.words_num > 0 else []) +
                      ([1.0] * (data.letters_num - 1) + [self.maskWeight(data)] if data.letters_num > 0 else []) +
                      [0.0] * (current_batch_length - data.words_num - data.letters_num)
                      for data in data_batch]
        # mask_array=np.array(mask_array)
        seq_len = [data.words_num + data.letters_num for data in data_batch]
        words_num = [data.words_num for data in data_batch]
        letters_num = [data.letters_num for data in data_batch]
        self.current_batch_index[bucket_id] += batch_size
        return input_array, output_array, mask_array, seq_len, current_batch_length, words_num, letters_num