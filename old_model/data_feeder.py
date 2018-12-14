from __future__ import absolute_import
import numpy as np
import random
from collections import namedtuple

from data_utility import DataUtility

Data = namedtuple('Data', ['in_data', 'words_num', 'letters_num', 'out_data'])

class DataFeederContext(object):
    def __init__(self, vocab_file_in_words="resource/vocab/vocab_in_words",
                 vocab_file_in_letters="resource/vocab/vocab_in_letters",
                 vocab_file_out="resource/vocab/vocab_out",
                 corpus_file_in_words="resource/train_data/train_in_ids_words",
                 corpus_file_in_letters="resource/train_data/train_in_ids_letters",
                 corpus_file_out="resource/train_data/train_out_ids",
                 max_sentence_length=30):
        # Use bucketing to reduce padding
        self.PAD_ID = 0

        self.data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words,
                                         vocab_file_in_letters=vocab_file_in_letters,
                                         vocab_file_out=vocab_file_out, max_sentence_length=max_sentence_length)

        corpus_in_words = self.load_corpus(corpus_file_in_words)
        corpus_in_letters = self.load_corpus(corpus_file_in_letters)
        corpus_out = self.load_corpus(corpus_file_out)
        self.all_data = []
        for i in range(len(corpus_in_words)):
            in_words_array = corpus_in_words[i].strip().split()
            in_letters_array = corpus_in_letters[i].strip().split()
            if len(in_letters_array) <= max_sentence_length:
                if len(in_words_array) + len(in_letters_array) <= max_sentence_length:
                    in_data = in_words_array + in_letters_array + [self.PAD_ID] * (max_sentence_length - len(in_words_array) - len(in_letters_array))
                    words_num = len(in_words_array)
                    letters_num = len(in_letters_array)
                else:
                    if len(in_letters_array) < max_sentence_length:
                        in_data = in_words_array[-(max_sentence_length - len(in_letters_array)):] + in_letters_array
                    else:
                        in_data = in_letters_array
                    words_num = max_sentence_length - len(in_letters_array)
                    letters_num = len(in_letters_array)
                out_data = corpus_out[i].strip()
                data = Data(in_data=in_data, words_num=words_num, letters_num=letters_num, out_data=out_data)
                self.all_data.append(data)
        self.num_samples = len(self.all_data)
        print ("samples num = " + str(self.num_samples))
        self.current_batch_index = 0
        self.max_sentence_length = max_sentence_length

    def load_corpus(self, corpus_file_in):
        corpus_array = []
        with open(corpus_file_in, mode="r") as f:
            for line in f:
                corpus_array.append(line)
        return corpus_array

    def maskWeight(self, data):
        in_data = data.in_data
        in_letters_id = in_data[data.words_num : data.words_num + data.letters_num]
        in_letters = [self.data_utility.id2token_in[int(id)] for id in in_letters_id]
        in_word = ''.join(in_letters)
        out_word = self.data_utility.id2token_out[int(data.out_data)]
        return 15.0 if in_word == out_word else 5.0

    def next_batch(self, batch_size=32):
        if self.current_batch_index + batch_size > self.num_samples:
            self.current_batch_index = 0
        if self.current_batch_index == 0:
            random.shuffle(self.all_data)
        start_index, end_index = self.current_batch_index, self.current_batch_index + batch_size
        data_batch = self.all_data[start_index:end_index]
        input_array = np.array([data.in_data for data in data_batch], dtype=np.int32)
        output_array = np.array([[self.PAD_ID] * max(data.words_num - 1, 0) +
                                 [data.out_data] * (data.letters_num + 1 if data.words_num > 0 else data.letters_num) +
                                 [self.PAD_ID] * (self.max_sentence_length - data.words_num - data.letters_num)
                                 for data in data_batch], dtype=np.int32)
        mask_array = [[0.0] * max(data.words_num - 1, 0) +
                      [1.0] * (data.letters_num + 1 if data.words_num > 0 else data.letters_num) +
                      [0.0] * (self.max_sentence_length - data.words_num - data.letters_num)
                      for data in data_batch]
        seq_len = [data.words_num + data.letters_num for data in data_batch]
        self.current_batch_index += batch_size
        return input_array, output_array, mask_array, seq_len

    def next_batch_fixmask(self, batch_size=32):
        if self.current_batch_index + batch_size > self.num_samples:
            self.current_batch_index = 0
        if self.current_batch_index == 0:
            random.shuffle(self.all_data)
        start_index, end_index = self.current_batch_index, self.current_batch_index + batch_size
        data_batch = self.all_data[start_index:end_index]
        input_array = np.array([data.in_data for data in data_batch], dtype=np.int32)
        output_array = np.array([[self.PAD_ID] * max(data.words_num - 1, 0) +
                                 [data.out_data] * (data.letters_num + 1 if data.words_num > 0 else data.letters_num) +
                                 [self.PAD_ID] * (self.max_sentence_length - data.words_num - data.letters_num)
                                 for data in data_batch], dtype=np.int32)
        mask_array = [[0.0] * max(data.words_num - 1, 0) +
                      ([1.0] if data.words_num > 0 else []) +
                      ([1.0] * (data.letters_num - 1) + [self.maskWeight(data)] if data.letters_num > 0 else []) +
                      [0.0] * (self.max_sentence_length - data.words_num - data.letters_num)
                      for data in data_batch]
        seq_len = [data.words_num + data.letters_num for data in data_batch]
        self.current_batch_index += batch_size
        return input_array, output_array, mask_array, seq_len

if __name__ == "__main__":
    df = DataFeederContext()
    input_array, output_array, mask_array, seq_len = df.next_batch_fixmask()
    print input_array
    print mask_array
    print seq_len

