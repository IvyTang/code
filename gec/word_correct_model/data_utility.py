#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

class DataUtility:
    def __init__(self, vocab_file_in_words=None, vocab_file_in_letters=None, vocab_file_out=None):

        self.start_str = "<start>"
        self.unk_str = "<unk>"
        self.num_str = "<num>"
        self.pun_str = "<pun>"
        self.pad_id = 0

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

            with open(vocab_file_in_letters, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in_letters[id] = token
                    self.token2id_in_letters[token] = id
            print ("in letters vocabulary size =", str(len(self.token2id_in_letters)))
            self.start_id = self.token2id_in_letters[self.start_str]
            print("in vocabulary size =", str(len(self.id2token_in_words) + len(self.id2token_in_letters)))
            self.in_letters_count = len(self.token2id_in_letters)

            with open(vocab_file_out, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_out[id] = token
                    self.token2id_out[token] = id
            self.out_words_count = len(self.token2id_out)
            print("out vocabulary size =", str(len(self.token2id_out)))

    def word2id(self, word):
        if re.match("^[a-zA-Z]$", word) or (word in self.token2id_in_words):
            word_out = word
        else:
            if re.match("^[+-]*[0-9]+.*[0-9]*$", word):
                word_out = self.num_str
            else:
                if re.match("^[^a-zA-Z0-9']*$", word):
                    word_out = self.pun_str
                else:
                    word_out = self.unk_str
        rid = self.token2id_in_words.get(word_out, -1)
        if rid == -1:
            return self.token2id_in_words[self.unk_str]
        return rid

    def words2ids(self, words):
        return [self.word2id(word) for word in words if len(word) > 0]

    def letters2ids(self, words):
        return [[self.start_id] + [self.token2id_in_letters.get(letter.lower(), self.token2id_in_letters[self.unk_str])
                                  for letter in letter_split if len(letter) > 0][:19] +
                [self.pad_id] * (19 - len(letter_split)) for letter_split in words]

    def outword2id(self, outword):
        return self.token2id_out.get(outword, self.token2id_out[self.unk_str])

    def ids2outwords(self, ids_out):
        return [self.id2token_out.get(id, self.unk_str) for id in ids_out]

    def ids2inwords(self, ids_in):
        return [self.id2token_in_words.get(int(id), self.unk_str) for id in ids_in]

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

    def sentence2ids(self, sentence):
        words_array = re.split('\\s+', sentence)
        words_num = len(words_array)
        letters_num = [len(letters) for letters in words_array]
        words_ids = self.words2ids(words_array)
        letters_ids = self.letters2ids(words_array)
        return words_ids, letters_ids, words_num, letters_num
