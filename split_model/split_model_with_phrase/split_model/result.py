#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import namedtuple

result = namedtuple('result', ['input_context', 'input_letters', 'output_words', 'res_words',
                               'res_probs', 'res_words_probs'])


class Result:

    def __init__(self):
        self.res_sentences = list()

    def find_last_index(self, string, str):
        last_position = -1
        while True:
            position = string.find(str, last_position + 1)
            if position == -1:
                return last_position
            last_position = position

    def parse_resul_pc(self, test_file):

        with open(test_file, "r") as f:
            for line in f:
                line = line.strip().split("|#|")
                if len(line) == 4:
                    input_context = ""
                    input_letters = ""
                    res_words = ""
                    res_probs = ""

                    if len(line[0].strip()) > 0:
                        input_context = line[0].split("\t")
                    if len(line[1].strip()) > 0:
                        input_letters = line[1].lower().split()

                    output_words = line[2].split("\t")
                    res_words_probs = line[3].split("\t")
                    if len(res_words_probs) == len(input_letters) + 1:
                        res_words = list()
                        res_probs = list()
                        for words_probs in res_words_probs:
                            words_probs = words_probs.split("|")
                            words = list()
                            probs = list()
                            for word_prob in words_probs:
                                idx = self.find_last_index(word_prob, ":")
                                if idx > 0:
                                    word = word_prob[:idx]
                                    prob = word_prob[idx+1:]
                                    prob = float(prob) if prob else 0.0
                                    words.append(word)
                                    probs.append(prob)
                                else:

                                    words.append(word_prob)
                                    probs.append(0.0)

                            res_words.append(words)
                            res_probs.append(probs)
                    else:
                        print("input letters length != res strings length: " + str(line))
                        continue

                    res_sentence = result(input_context=input_context, input_letters=input_letters,
                                          output_words=output_words, res_words=res_words, res_probs=res_probs,
                                          res_words_probs=res_words_probs)
                    if len(input_letters) + len(input_context) > 0:
                        self.res_sentences.append(res_sentence)
                else:
                    print("split line String error : " + str(line))
        f.close()

    def parse_resul_android(self, test_file):

        return

