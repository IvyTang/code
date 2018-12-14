#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import nltk
import numpy as np


class DataUtility:
    def __init__(self, vocab_file_in_words=None, vocab_file_in_letters=None, vocab_file_out=None, emoji_file="emojis"):

        self.start_str = "<start>"
        self.bos_str = "<bos>"
        self.eos_str = "<eos>"
        self.unk_str = "<unk>"
        self.num_str = "<num>"
        self.pun_str = "<pun>"
        self.emoji_str = "<emoji>"
        self.emoji_dict = {}
        self.fullvocab_set = None
        self.pad_id = 0
        self.stem = nltk.stem.SnowballStemmer('english')

        if vocab_file_in_words and vocab_file_in_letters and vocab_file_out:
            self.id2token_in_words, self.id2token_in_letters, self.id2token_out = {}, {}, {}
            self.token2id_in_words, self.token2id_in_letters, self.token2id_out = {}, {}, {}
            with open(vocab_file_in_words, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in_words[id] = token
                    self.token2id_in_words[token] = id
            print ("in words vocabulary size =", str(len(self.token2id_in_words)))
            self.in_words_count = len(self.token2id_in_words)
            self.bos_id = self.token2id_in_words[self.bos_str]
            self.eos_id = self.token2id_in_words[self.eos_str]

            with open(vocab_file_in_letters, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in_letters[id] = token
                    self.token2id_in_letters[token] = id

            print("in letters vocabulary size =", str(len(self.token2id_in_letters)))
            self.start_id = self.token2id_in_letters[self.start_str]
            print("in vocabulary size =", str(len(self.id2token_in_words) + len(self.id2token_in_letters)))
            self.in_letters_count = len(self.token2id_in_letters)

            with open(vocab_file_out, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_out[id] = token
                    self.token2id_out[token] = id
            print("out vocabulary size =", str(len(self.token2id_out)))
            self.out_words_count = len(self.token2id_out)

            with open(emoji_file, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("\t")
                    id = int(id)
                    self.emoji_dict[token] = id

            print("emoji vocabulary size =", str(len(self.emoji_dict)))

    def softmax(self, logits):
        exp_logits = np.exp(logits)
        exp_sum = np.expand_dims(np.sum(exp_logits, -1), -1)
        return exp_logits / exp_sum

    def word2id(self, word):
        if re.match("^[a-zA-Z]$", word) or (word in self.token2id_in_words):
            word_out = word
        elif word in self.emoji_dict:
            word_out = self.emoji_str

        elif re.match("^[+-]*[0-9]+.*[0-9]*$", word):
            word_out = self.num_str
        elif re.match("^[^a-zA-Z0-9']*$", word):
            word_out = self.pun_str
        else:
            word_out = self.unk_str

        rid = self.token2id_in_words.get(word_out, -1)
        if rid == -1:
            return self.token2id_in_words[self.unk_str]
        return rid

    def words2ids(self, words):
        return [self.bos_id] + [self.word2id(word) for word in words if len(word) > 0] + [self.eos_id]

    def letters2ids(self, words):
        max_length = 20
        return [[self.token2id_in_letters.get(letter.lower(), self.token2id_in_letters[self.unk_str])
                                  for letter in letter_split if len(letter) > 0][:max_length] +
                [self.pad_id] * (max_length - len(letter_split)) for letter_split in words]

    def ids2inwords(self, ids_in):
        return [self.id2token_in_words.get(int(id), self.unk_str) for id in ids_in]

    def ids2outwords(self, ids_in):
        return [self.id2token_out.get(int(id), self.unk_str) for id in ids_in]

    def data2ids_line(self, data_line):
        data_line_split = re.split("\\|#\\|", data_line)
        letters_line = data_line_split[0].replace(" ","").split("\t")
        raw_words_line = data_line_split[1].strip().split("\t")
        words_line = []
        for i in range(len(raw_words_line)):
            # if raw_words_line[i].lower() != letters_line[i]:
            #     words_line.append(letters_line[i])
            # else:
            words_line.append(raw_words_line[i])
        words_ids = self.words2ids(words_line)
        letters_ids = self.letters2ids(letters_line)
        words_num = len(words_line)
        letters_num = [len(letter) for letter in letters_line]
        return raw_words_line, words_line, letters_line, words_ids, letters_ids, words_num, letters_num

    def data2ids_line_phase2(self, data_line):
        data_line_split = re.split("\\|#\\|", data_line)
        letters_line = data_line_split[0].replace(" ", "").split("\t")
        words_line = data_line_split[1].split("\t")
        raw_letters_line = data_line_split[2].split("\t")
        outputs_line = data_line_split[3].strip().split("\t")
        words_ids = self.words2ids(words_line)
        letters_ids = self.letters2ids(letters_line)
        words_num = len(words_line)
        letters_num = [len(letter) for letter in letters_line]
        return outputs_line, words_line, raw_letters_line, words_ids, letters_ids, words_num, letters_num

    def sentence2ids(self, sentence):
        words_array = re.split('\\s+', sentence.strip())
        words_array_stem = [(word if len(self.stem.stem(word)) == len(word) else self.stem.stem(word)) for word in
                            words_array]
        words_num = len(words_array)
        letters_num = [len(letters) for letters in words_array]
        words_ids = self.words2ids(words_array)
        letters_ids = self.letters2ids(words_array)
        return words_array, words_ids, letters_ids, words_num, letters_num
