#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import spacy
nlp = spacy.load("en")
from spacy_lemmatizer import Lemmatizer


class DataUtility:
    def __init__(self, vocab_file_in_words=None, vocab_file_in_letters=None, vocab_file_out=None,
                 vocab_file_lemma=None, vocab_freq_file=None, emoji_file="emojis"):

        self.start_str = "<start>"
        self.unk_str = "<unk>"
        self.num_str = "<num>"
        self.pun_str = "<pun>"
        self.bos_str = "<bos>"
        self.eos_str = "<eos>"
        self.emoji_str = "<emoji>"
        self.emoji_dict = {}
        self.lemma_words_lists = []
        self.pun_list = [".",",","?","!",":",";"]
        self.fullvocab_set = None
        self.pad_id = 0
        self.lemmatizer = Lemmatizer()
        self.vocab_freq_dict = {}

        if vocab_file_in_words and vocab_file_in_letters and vocab_file_out and vocab_freq_file:
            self.id2token_in_words, self.id2token_in_letters, self.id2token_out = {}, {}, {}
            self.token2id_in_words, self.token2id_in_letters, self.token2id_out = {}, {}, {}
            with open(vocab_file_in_words, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in_words[id] = token
                    self.token2id_in_words[token] = id
            print("in words vocabulary size =", str(len(self.token2id_in_words)))
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
            print("in vocabulary size =", str(len(self.id2token_in_words)+len(self.id2token_in_letters)))
            self.in_letters_count = len(self.token2id_in_letters)

            with open(vocab_file_out, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_out[id] = token
                    self.token2id_out[token] = id
            self.out_words_count = len(self.token2id_out)
            print("out vocabulary size =", str(len(self.token2id_out)))

            with open(emoji_file, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("\t")
                    id = int(id)
                    self.emoji_dict[token] = id

            with open(vocab_file_lemma, mode="r") as f:
                for line in f:
                    tokens = line.strip().split("\t")
                    self.lemma_words_lists.append(tokens)

            print("emoji vocabulary size =", str(len(self.emoji_dict)))

            with open(vocab_freq_file, mode="r") as f:
                for line in f:
                    token, freq = line.strip().split("##")
                    self.vocab_freq_dict[token] = float(freq)

            print("vocab freq size =", str(len(self.vocab_freq_dict)))

    def softmax(self, logits):
        exp_logits = np.exp(logits)
        exp_sum = np.expand_dims(np.sum(exp_logits, -1), -1)
        return exp_logits / exp_sum

    def word2id(self, word):
        if re.match("^[a-zA-Z]$", word) or (word in self.token2id_in_words):
            word_out = word
        elif word.lower() in self.token2id_in_words:
            word_out = word.lower()
        elif word.lower().capitalize() in self.token2id_in_words:
            word_out = word.lower().capitalize()
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

    def first_word2id(self, word):
        if word.lower().capitalize() in self.token2id_in_words:
            word_out = word.lower().capitalize()
        elif word in self.token2id_in_words:
            word_out = word
        elif word.lower() in self.token2id_in_words:
            word_out = word.lower()
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
        return [[self.token2id_in_letters.get(letter.lower(), self.token2id_in_letters[self.unk_str]) if letter not in
                self.pun_list else self.token2id_in_letters.get(letter, self.token2id_in_letters[self.unk_str])
                for letter in letter_split if len(letter) > 0][:max_length] + [self.pad_id] *
                (max_length - len(letter_split)) for letter_split in words]

    def ids2inwords(self, ids_in):
        return [self.id2token_in_words.get(int(id), self.unk_str) for id in ids_in]

    def outword2id(self, outword):
        return self.token2id_out.get(outword, self.token2id_out[self.unk_str])

    def ids2outwords(self, ids_out):
        return [self.id2token_out.get(int(id), self.unk_str) for id in ids_out]

    def data2ids_line_cnn(self, data_line):
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

    def data2ids_line(self, data_line):
        data_line_split = re.split("\\|#\\|", data_line)
        letters_line = data_line_split[0].replace(" ","").split("\t")
        output_line = data_line_split[1].strip().split("\t")
        words_line = []
        for i in range(len(output_line)):
            if output_line[i].lower() != letters_line[i]:
                words_line.append(letters_line[i])
            else:
                words_line.append(output_line[i])

        words_line = " ".join(words_line)
        output_line = " ".join(output_line)
        sentence_original, _, _, _ = self.lemmatizer.lemmatize(output_line)
        _, sentence_lemmatized, _, lemmatized_idx = self.lemmatizer.lemmatize(words_line)
        if len(sentence_original) != len(words_line.split()) or len(sentence_original) != len(sentence_lemmatized):
            return None

        words_num = len(sentence_original)

        words_ids = self.words2ids(sentence_original)
        lemma_words_ids = self.words2ids(sentence_lemmatized)[1:-1]
        letters_ids = self.letters2ids(letters_line)

        return sentence_original, sentence_lemmatized, letters_line, words_ids, lemma_words_ids, letters_ids, \
               words_num, lemmatized_idx

    def sentence2ids(self, sentence):

        sentence_original, sentence_lemmatized, words_pos, lemmatized_idx = self.lemmatizer.lemmatize(sentence)
        assert len(sentence_original) == len(sentence_lemmatized)
        # words_num = len(sentence_original)
        # sentence_original = sentence.strip().split()
        # letters_num = [len(letters) for letters in sentence_lemmatized]
        words_ids = self.words2ids(sentence_original)
        lemma_words_ids = self.words2ids(sentence_lemmatized)[1:-1]
        lemma_words_ids.extend([0, 0])
        letters_ids = self.letters2ids(sentence_original)

        return sentence_original, sentence_lemmatized, words_pos, words_ids, lemma_words_ids, letters_ids, lemmatized_idx

    def sentence2ids_no_token(self, sentence):
        words = sentence.strip().split()
        words_lemmatized = []
        lemmatized_idx = []
        words_pos = []

        for i in range(len(words)):
            word = words[i]
            word_original, word_lemmatized, word_pos, idx = self.lemmatizer.lemmatize(word)
            if len(idx) > 0 and len(word_original) == 1 and len(word_lemmatized) == 1 and len(word_pos) == 1:
                words_lemmatized.append(word_lemmatized[0])
                lemmatized_idx.append(i)
            else:
                words_lemmatized.append("<unk>")
            words_pos.append(word_pos[0])

        words_ids = self.words2ids(words)
        lemma_words_ids = self.words2ids(words_lemmatized)[1:-1]
        lemma_words_ids.extend([0, 0])
        letters_ids = self.letters2ids(words)

        return words, words_lemmatized, words_pos, words_ids, lemma_words_ids, letters_ids, lemmatized_idx

    def flatten(self, lst):
        # Assume lst is a list of lists, whose every item is an ID list
        # This function flatten lst into a list of IDs.
        # e.g.: [[1, 2], [3, 4, 5, 6], [7, 8, 9]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9]
        return [x for item in lst for x in item]

    def real_word_edits1(self, word):
        """与'word'的编辑距离为1，且在词表里的全部结果"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        correct_word = word
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [L + R[1:] for L, R in splits if R]
        deletes = [word for word in deletes if word in self.token2id_in_words and word != correct_word]

        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        transposes = [word for word in transposes if word in self.token2id_in_words and word != correct_word]

        replaces = [L + c + R[1:] for L, R in splits for c in letters if len(R) > 0]
        replaces = [word for word in replaces if word in self.token2id_in_words and word != correct_word]

        inserts = [L + c + R for L, R in splits for c in letters]
        inserts = [word for word in inserts if word in self.token2id_in_words and word != correct_word]

        cap_or_til = []
        cap = correct_word.lower().capitalize()
        upper = correct_word.upper()
        if cap != correct_word and cap in self.token2id_in_words:
            cap_or_til.append(cap)
        if upper != correct_word and upper in self.token2id_in_words:
            cap_or_til.append(upper)

        real_word_edit1_lists = [deletes, transposes, replaces, inserts, cap_or_til]
        while [] in real_word_edit1_lists:
            real_word_edit1_lists.remove([])

        if len(real_word_edit1_lists) == 0:
            return None
        else:
            return [self.token2id_in_words[correct_word]] + [self.token2id_in_words[word] for word in self.flatten(real_word_edit1_lists)]

    def lemma_word_set(self, word):
        """与'word'为lemma关系，且在词表里的全部结果"""
        for lemma_words_list in self.lemma_words_lists:
            if word in lemma_words_list:
                return lemma_words_list,\
                       [self.token2id_in_words[lemma_word] for lemma_word in lemma_words_list]
        return None
