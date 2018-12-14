#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
from collections import namedtuple
import spacy
from matplotlib import pyplot as plt
Token = namedtuple('Token', ['text', 'lemma_', 'pos_', 'tag_'])

lemmatizer = spacy.load('en')


# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    plt.hist(myList)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()


def post_process_tokens(tokens):
    tokens_processed = []
    for token in tokens:
        token_my = Token(text=token.text, lemma_=token.lemma_, pos_=token.pos_, tag_=token.tag_)
        tokens_processed.append(token_my)
    i = 1
    while i < len(tokens_processed) - 1:
        token = tokens_processed[i]
        token_pred = tokens_processed[i - 1]
        token_succ = tokens_processed[i + 1]
        if (token.text == 'num' or token.text == 'pun') and token_pred.text == '<' and token_succ.text == '>':
            del tokens_processed[i + 1]
            text = '<' + token.text + '>'
            token_my = Token(text=text, lemma_=text, pos_=text, tag_=text)
            tokens_processed[i] = token_my
            del tokens_processed[i - 1]
            i -= 1
        i += 1
    return tokens_processed


def post_process_word(original_word, lemmatized_word):
    if lemmatized_word == '-PRON-':
        return original_word
    if original_word.isupper():
        return lemmatized_word.upper()
    if original_word.istitle():
        return lemmatized_word.capitalize()
    return lemmatized_word


def get_lemmatized_idx(tokens):
    lemmatized_idx_array = []
    for i in range(len(tokens)):
        token = tokens[i]
        if token.pos_ in {'VERB', 'ADJ', 'ADV', 'NOUN'}:
            lemmatized_idx_array.append(int(i))
        else:
            if token.text.lower() != token.lemma_.lower() and token.lemma_ != '-PRON-':
                print(token.text, token.lemma_, token.pos_, token.tag_)
    return lemmatized_idx_array


def lemmatize(sentence):
    sentence = sentence.strip()
    tokens = lemmatizer(sentence)
    tokens_processed = post_process_tokens(tokens)
    words_original = [token.text for token in tokens_processed]
    words_lemmatized = [post_process_word(token.text, token.lemma_) for token in tokens_processed]
    lemmatized_idx_array = get_lemmatized_idx(tokens_processed)
    return words_original, words_lemmatized, lemmatized_idx_array


def InputEngineTest(test_file_native, test_file_result):

    total_count, native_correct_count = 0.0, 0.0

    with open(test_file_native, "r") as f_native, open(test_file_result, "r") as f_result:
        for (line_native, line_result) in zip(f_native, f_result):
            result_original, _, _ = lemmatize(line_result)
            line_result = " ".join(result_original)
            line_native = line_native.strip()
            if len(line_native.split()) != len(line_result.split()):
                continue
            total_count += 1

            if line_native.lower() == line_result.lower():
                native_correct_count += 1
            else:
                print("gec: " + line_native + "\n" + "tgt: " + line_result)
                gec_words = line_native.split()
                tgt_words = line_result.split()
                for (gec_word, tgt_word) in zip(gec_words, tgt_words):
                    if gec_word.lower() != tgt_word.lower():
                        print(gec_word + " -> " + tgt_word)

    print("precious:", float(native_correct_count / total_count))
    print(native_correct_count, total_count)


if __name__ == "__main__":

    args = sys.argv
    test_file_native = args[1]
    test_file_result = args[2]

    InputEngineTest(test_file_native, test_file_result)

