#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import re
import os
import random


class TrainDataProducer:

    def __init__(self):

        self.word_regex = "^[a-zA-Z']+$"
        self.num_regex = "^[+-]*[0-9]+.*[0-9]*$"
        self.pun_regex = "^[^a-zA-Z0-9']*$"

        self.vocab_split_flag = "##"
        self.pad_flag = "_PAD"
        self.eos_flag = "<eos>"
        self.num_flag = "<num>"
        self.pun_flag = "<pun>"
        self.emoji_flag = "<emoji>"
        self.unk_flag = "<unk>"
        self.und_flag = "<und>"
        self.start_flag = "<start>"
        self.not_phrase_flag = "<unp>"
        self.unk_emoji_flag = "<unk_emoji>"
        self.line_split_flag = "#"

        self.unk_emoji_id = 0

        self.pad_id = 0
        self.in_eos_id = 1
        self.in_unk_id = 2
        self.num_id = 3
        self.pun_id = 4
        self.emoji_id = 5

        self.letter_unk_id = 0
        self.letter_start_id = 1

        self.out_eos_id = 0
        self.out_emoji_id = 1
        self.out_unk_id = 2
        self.und_id = 3

        self.not_phrase_id = 1

        self.emojis_dict = dict()
        self.emojis_combine_dict = dict()
        self.phrase_dict = dict()
        self.full_words_dict = dict()

        self.big_words_set = set()
        self.dict_letters_set = set()
        self.data_letters_set = set()
        self.data_in_words_set = set()
        self.data_out_words_set = set()

        self.rnn_vocab_dict = dict()
        self.words_count_dict_from_data = dict()
        self.phrase_count_dict_from_data = dict()
        self.phrase_dict_from_data = dict()

        self.in_word_id_dict = dict()
        self.letter_id_dict = dict()
        self.out_word_id_dict = dict()
        self.phrase_id_dict = dict()
        self.emoji_id_dict = dict()

        self.line_num = 0

    def load_vocab(self, vocab_file, split_flag, vocab_type):
        vocab_dict = dict()
        with open(vocab_file, "r") as f:
            for line in f:
                line = line.strip()
                line_split = re.split(split_flag, line)
                if len(line_split) == 2:
                    token, id = line_split
                    vocab_dict[token] = int(id)
                elif len(line_split) == 1:
                    token = line_split[0]
                    vocab_dict[token] = len(vocab_dict)
                else:
                    print(vocab_type + " vocab split error : " + line)
        f.close()
        print(vocab_type + " vocab num = " + str(len(vocab_dict)))
        return vocab_dict

    def is_word_or_emoji(self, word):
        if (len(word) > 0 and re.match(self.word_regex, word)) or word in self.emojis_dict:
            return True
        else:
            return False

    def is_word(self, word):
        if len(word) > 0 and re.match(self.word_regex, word):
            return True
        else:
            return False

    def to_emoji_combine(self, emoji_1, emoji_2):
        if (emoji_1 + emoji_2) in self.emojis_combine_dict:
            return emoji_1 + emoji_2
        elif (emoji_2 + emoji_1) in self.emojis_combine_dict:
            return emoji_2 + emoji_1
        return ""

    def convert_words_to_emoji_combine_out(self, words):
        res = [i for i in range(len(words))]

        res_idx = 0
        prev_emoji = ""
        emoji_combine = ""

        for i in range(len(words)):
            cur_word = words[i]

            if cur_word not in self.emojis_dict:
                if emoji_combine != "":
                    while res_idx < i:
                        res[res_idx] = emoji_combine
                        res_idx += 1
                elif prev_emoji != "":
                    while res_idx < i:
                        res[res_idx] = prev_emoji
                        res_idx += 1
                prev_emoji = ""
                emoji_combine = ""
                res[res_idx] = self.unk_emoji_flag
                res_idx += 1

            elif prev_emoji == "":
                prev_emoji = cur_word

            elif cur_word in emoji_combine:
                res[res_idx] = emoji_combine
                res_idx += 1

            elif prev_emoji == cur_word:
                continue

            elif emoji_combine == "":
                emoji_combine = self.to_emoji_combine(prev_emoji, cur_word)
                if emoji_combine == "":
                    while res_idx < i:
                        res[res_idx] = prev_emoji
                        res_idx += 1
                    prev_emoji = cur_word
                else:
                    while res_idx <= i:
                        res[res_idx] = emoji_combine
                        res_idx += 1

            else:
                emoji_combine = ""
                prev_emoji = cur_word

        while res_idx < len(words):
            res[res_idx] = self.unk_emoji_flag if prev_emoji == "" else prev_emoji
            res_idx += 1

        return res

    def calc_words_phrase_freq_line(self, line):
        words = re.split(self.line_split_flag, line)
        if len(words) > 0:

            for word in words:
                if self.is_word_or_emoji(word):
                    if word in self.words_count_dict_from_data:
                        self.words_count_dict_from_data[word] += 1
                    else:
                        self.words_count_dict_from_data[word] = 1

            for i in range(len(words)):
                if i + 1 < len(words):
                    if self.is_word(words[i]) and self.is_word(words[i + 1]):
                        phrase_2 = words[i] + " " + words[i + 1]
                        if phrase_2 in self.phrase_count_dict_from_data:
                            self.phrase_count_dict_from_data[phrase_2] += 1
                        else:
                            self.phrase_count_dict_from_data[phrase_2] = 1
                if i + 2 < len(words):
                    if self.is_word(words[i]) and self.is_word(words[i + 1]) and self.is_word(words[i + 2]):
                        phrase_3 = words[i] + " " + words[i + 1] + " " + words[i + 2]
                        if phrase_3 in self.phrase_count_dict_from_data:
                            self.phrase_count_dict_from_data[phrase_3] += 1
                        else:
                            self.phrase_count_dict_from_data[phrase_3] = 1
        else:
            print("split error :", line)

        return

    def calc_words_freq_all_data(self, data_path_in):
        root_path = data_path_in
        files_list = os.listdir(root_path)
        for file in files_list:
            file_path = os.path.join(root_path, file)
            print(file_path)
            with open(file_path) as f:
                for line in f:
                    letters_line, words_line = line.rstrip().split("\t")
                    self.line_num += 1
                    self.calc_words_phrase_freq_line(words_line)

            f.close()
        print("words count dict from data size =", len(self.words_count_dict_from_data))
        print("words count dict from data =", self.words_count_dict_from_data)
        print("lineNum =", self.line_num)

        return

    def combine_words_and_phrase_dict(self, words_num, phrase_num):

        sorted_words_count_from_data = sorted(self.words_count_dict_from_data.items(),key=lambda x:x[1],reverse=True)

        for (word, count) in sorted_words_count_from_data:
            if word in self.full_words_dict:
                if len(self.rnn_vocab_dict) < words_num:
                    self.rnn_vocab_dict[word] = count
                    if word not in self.emojis_dict:
                        for char in word:
                            self.dict_letters_set.add(char)

        for phrase, count in self.phrase_count_dict_from_data.items():
            phrase_split = phrase.split()
            if len(phrase_split) == 3:
                phrase_2 = " ".join(phrase_split[:2])
                if phrase_2 in self.phrase_count_dict_from_data:
                    self.phrase_count_dict_from_data[phrase_2] -= count

        sorted_phrase_count_from_data = sorted(self.phrase_count_dict_from_data.items(),key=lambda x:x[1],reverse=True)
        for (phrase, count) in sorted_phrase_count_from_data:
            save_phrase = True
            phrase_split = phrase.split()
            for word in phrase_split:
                if word not in self.rnn_vocab_dict:
                    save_phrase = False
                    break
            if save_phrase:
                if len(self.phrase_dict_from_data) < phrase_num:
                    self.phrase_dict_from_data[phrase] = count

        for word in self.full_words_dict:
            if word not in self.rnn_vocab_dict:
                self.big_words_set.add(word)

        print("rnn vocab dict size =", len(self.rnn_vocab_dict))
        print("rnn vocab dict =", self.rnn_vocab_dict)
        print("big words set size =", len(self.big_words_set))
        print("dict letters set size =", len(self.dict_letters_set))
        print("dict letters set =", self.dict_letters_set)
        print("phrase dict from data size =", len(self.phrase_dict_from_data))

        return

    def convert_in_words(self, words, rate_threshold):
        word_num = 0
        unknown_word_num = 0

        words_converted_list = []

        for word in words:
            if word in self.rnn_vocab_dict:
                word_converted = word
                word_num += 1
            elif word.lower() in self.rnn_vocab_dict:
                word_converted = word.lower()
                word_num += 1
            elif word in self.emojis_dict:
                word_converted = self.emoji_flag
                word_num += 1
            elif re.match(self.num_regex, word):
                word_converted = self.num_flag
                unknown_word_num += 1
            elif re.match(self.pun_regex, word):
                word_converted = self.pun_flag
                unknown_word_num += 1
            else:
                word_converted = self.unk_flag
                unknown_word_num += 1
            words_converted_list.append(word_converted)

        word_rate = float(word_num / (word_num + unknown_word_num))
        if word_rate >= rate_threshold and len(words_converted_list) > 0:
            return words_converted_list
        else:
            return None

    def convert_out_words(self, words, rate_threshold):
        word_num = 0
        unknown_word_num = 0

        words_converted_list = []

        for word in words:
            if word in self.rnn_vocab_dict:
                word_converted = word
                word_num += 1
            elif word.lower() in self.rnn_vocab_dict:
                word_converted = word.lower()
                word_num += 1
            elif word in self.big_words_set and word not in self.emojis_dict:
                word_converted = self.und_flag
                word_num += 1
            else:
                word_converted = self.unk_flag
                unknown_word_num += 1
            words_converted_list.append(word_converted)

        word_rate = float(word_num / (word_num + unknown_word_num))
        if word_rate >= rate_threshold:
            return words_converted_list
        else:
            return None

    def convert_letters(self, letters_list):

        letters_converted_list = []
        for letters in letters_list:
            if len(letters) > 0:
                letter_converted_list = []
                for letter in letters:
                    if letter in self.dict_letters_set:
                        letter_converted_list.append(letter)
                    else:
                        letter_converted_list.append(self.unk_flag)
                letters_converted_list.append(letter_converted_list)
            else:
                letters_converted_list.append("")
        return letters_converted_list

    def convert_line(self, line, rate_threshold, file_writer, is_train):
        letters_list = []
        words_list = []

        line_split = re.split("\t", line)
        if len(line_split) == 2:
            letters, words = line_split
            letters_list = letters.split(self.line_split_flag)
            words_list = words.split(self.line_split_flag)
        else:
            print("line split error :", line)

        if len(words_list) == len(letters_list):

            if is_train:
                in_words_converted = self.convert_in_words(words_list, rate_threshold)
                out_words_converted = self.convert_out_words(words_list, rate_threshold)
                letters_converted = self.convert_letters(letters_list)

                if in_words_converted is not None and out_words_converted is not None:
                    for letters in letters_converted:
                        for letter in letters:
                            if len(letter) > 0: self.data_letters_set.add(letter)

                    for word in in_words_converted:
                        if len(word) > 0: self.data_in_words_set.add(word)

                    for word in out_words_converted:
                        if len(word) > 0: self.data_out_words_set.add(word)

                    file_writer.write("\t".join(letters_list) + "|#|" + "\t".join(words_list) + "\n")

            else:
                file_writer.write("\t".join(letters_list) + "|#|" + "\t".join(words_list) + "\n")

        return

    def convert_data(self, data_path_in, data_path_out, rate_threshold,
                           train_data_num, dev_data_num, test_data_num):
        root_path = data_path_in
        files_list = os.listdir(root_path)
        train_rate = float(train_data_num / self.line_num)
        dev_rate = float(dev_data_num / self.line_num)
        test_rate = float(test_data_num / self.line_num)

        if not os.path.isdir(data_path_out):
            os.makedirs(data_path_out)

        train_writer = open(os.path.join(data_path_out, "train_data"), "w")
        dev_writer = open(os.path.join(data_path_out, "dev_data"), "w")
        test_writer = open(os.path.join(data_path_out, "test_data"), "w")

        for file in files_list:
            file_path = os.path.join(root_path, file)
            print(file_path)
            with open(file_path) as f:
                for line in f:
                    line = line.rstrip()
                    rand = random.random()
                    if rand < train_rate:
                        self.convert_line(line, rate_threshold, train_writer, True)
                    elif rand < train_rate + dev_rate:
                        self.convert_line(line, rate_threshold, dev_writer, False)
                    elif rand < train_rate + dev_rate + test_rate:
                        self.convert_line(line, rate_threshold, test_writer, False)
            f.close()

        train_writer.close()
        dev_writer.close()
        test_writer.close()

        print("data words set size =", len(self.data_in_words_set))
        print("data words set =", self.data_in_words_set)
        print("data letters set size =", len(self.data_letters_set))
        print("data letters set =", self.data_letters_set)

        return

    def make_vocab_map(self):

        self.in_word_id_dict[self.pad_flag] = self.pad_id
        self.in_word_id_dict[self.eos_flag] = self.in_eos_id
        self.in_word_id_dict[self.unk_flag] = self.in_unk_id
        self.in_word_id_dict[self.num_flag] = self.num_id
        self.in_word_id_dict[self.pun_flag] = self.pun_id
        self.in_word_id_dict[self.emoji_flag] = self.emoji_id

        self.letter_id_dict[self.unk_flag] = self.letter_unk_id
        self.letter_id_dict[self.start_flag] = self.letter_start_id

        self.out_word_id_dict[self.eos_flag] = self.out_eos_id
        self.out_word_id_dict[self.emoji_flag] = self.out_emoji_id
        self.out_word_id_dict[self.unk_flag] = self.out_unk_id
        self.out_word_id_dict[self.und_flag] = self.und_id

        self.phrase_id_dict[self.pad_flag] = self.pad_id
        self.phrase_id_dict[self.not_phrase_flag] = self.not_phrase_id

        self.emoji_id_dict[self.unk_emoji_flag] = self.unk_emoji_id

        id = len(self.letter_id_dict)
        data_letters_list = list(self.data_letters_set)
        data_letters_list.sort()
        for letter in data_letters_list:
            if letter not in self.letter_id_dict:
                self.letter_id_dict[letter] = id
                id += 1

        print("letter id dict size =", len(self.letter_id_dict))
        print("letter id dict =", self.letter_id_dict)

        id = len(self.in_word_id_dict)
        for word in self.rnn_vocab_dict:
            if word in self.data_in_words_set:
                if word not in self.in_word_id_dict:
                    self.in_word_id_dict[word] = id
                    id += 1

        print("in word id dict size =", len(self.in_word_id_dict))
        print("in word id dict =", self.in_word_id_dict)

        id = len(self.out_word_id_dict)
        for word in self.rnn_vocab_dict:
            if word in self.data_out_words_set and word not in self.emojis_dict:
                if word not in self.out_word_id_dict:
                    self.out_word_id_dict[word] = id
                    id += 1

        print("out word id dict size =", len(self.out_word_id_dict))
        print("out word id dict =", self.out_word_id_dict)

        id = len(self.phrase_id_dict)

        phrase_dict_to_save = self.phrase_dict if len(self.phrase_dict) > 0 else self.phrase_dict_from_data

        for phrase in phrase_dict_to_save:
            if phrase == self.pad_flag or phrase == self.not_phrase_flag:
                continue
            save_phrase = True
            phrase_split = phrase.split()
            for word in phrase_split:
                if word not in self.in_word_id_dict or word not in self.out_word_id_dict:
                    save_phrase = False
                    break
            if save_phrase:
                self.phrase_id_dict[phrase] = id
                id += 1

        print("phrase id dict size =", len(self.phrase_id_dict))
        print("phrase id dict =", self.phrase_id_dict)

        id = len(self.emoji_id_dict)
        for word in self.rnn_vocab_dict:
            if word in self.data_out_words_set and word in self.emojis_dict:

                self.emoji_id_dict[word] = id
                id += 1

        for emojis_combine, _ in sorted(self.emojis_combine_dict.items(),key=lambda x:x[1]):

            self.emoji_id_dict[emojis_combine] = id
            id += 1

        print("emojis id dict size =", len(self.emoji_id_dict))
        print("emojis id dict =", self.emoji_id_dict)

        return

    def save_map_to_file(self, id_map, vocab_file_out):
        with open(vocab_file_out, "w") as f:
            for word, id in id_map.items():
                f.write(word + self.vocab_split_flag + str(id) + "\n")
        f.close()

        return

    def save_vocab_files(self, data_path_out):
        self.make_vocab_map()
        self.save_map_to_file(self.in_word_id_dict, data_path_out + "/vocab_in_words")
        self.save_map_to_file(self.out_word_id_dict, data_path_out + "/vocab_out")
        self.save_map_to_file(self.letter_id_dict, data_path_out + "/vocab_in_letters")
        self.save_map_to_file(self.phrase_id_dict, data_path_out + "/vocab_phrase")
        self.save_map_to_file(self.emoji_id_dict, data_path_out + "/vocab_out_emoji")

        return

    def convert_in_words_ids(self, words):

        if len(words) >= 2:
            ids_list = [self.in_word_id_dict[self.eos_flag]]
            for word in words:
                if word in self.in_word_id_dict:
                    ids_list.append(self.in_word_id_dict[word])
                elif word.lower() in self.in_word_id_dict:
                    ids_list.append(self.in_word_id_dict[word.lower()])
                elif word in self.emojis_dict:
                    ids_list.append(self.in_word_id_dict[self.emoji_flag])
                elif re.match(self.num_regex, word):
                    ids_list.append(self.in_word_id_dict[self.num_flag])
                elif re.match(self.pun_regex, word):
                    ids_list.append(self.in_word_id_dict[self.pun_flag])
                else:
                    ids_list.append(self.in_word_id_dict[self.unk_flag])

            str_ids = " ".join([str(id) for id in ids_list])
            return str_ids
        else:
            return None

    def convert_out_words_ids(self, words):

        if len(words) >= 2:
            ids_list = [self.out_word_id_dict[self.eos_flag]]
            for word in words:
                if word in self.out_word_id_dict:
                    ids_list.append(self.out_word_id_dict[word])
                elif word.lower() in self.out_word_id_dict:
                    ids_list.append(self.out_word_id_dict[word.lower()])
                elif word in self.big_words_set and word not in self.emojis_dict:
                    ids_list.append(self.out_word_id_dict[self.und_flag])
                elif word in self.emojis_dict:
                    ids_list.append(self.out_word_id_dict[self.emoji_flag])
                else:
                    ids_list.append(self.out_word_id_dict[self.unk_flag])

            str_ids = " ".join([str(id) for id in ids_list])
            return str_ids
        else:
            return None

    def convert_letters_ids(self, letters_list):

        if len(letters_list) >= 2:
            str_letters_ids_list = [str(self.letter_id_dict[self.start_flag])]
            for letters in letters_list:
                letter_ids_list = [self.letter_id_dict[self.start_flag]]
                if len(letters) > 0 and letters not in self.emojis_dict:
                    for letter in letters:
                        if letter in self.letter_id_dict:
                            letter_ids_list.append(self.letter_id_dict[letter])
                        else:
                            letter_ids_list.append(self.letter_id_dict[self.unk_flag])
                str_letter_ids = " ".join([str(id) for id in letter_ids_list])
                str_letters_ids_list.append(str_letter_ids)
            str_letters_ids = "#".join(str_letters_ids_list)

            return str_letters_ids
        else:
            return None

    def convert_phrase_ids(self, words):

        if len(words) >= 2:
            ids_list = [self.phrase_id_dict[self.pad_flag]]
            for i in range(len(words)):
                if i + 1 < len(words):
                    phrase_2 = words[i] + " " + words[i+1]
                    if phrase_2 in self.phrase_id_dict:
                        ids_list.append(self.phrase_id_dict[phrase_2])
                        continue
                if i + 2 < len(words):
                    phrase_3 = words[i] + " " + words[i+1] + " " + words[i+2]
                    if phrase_3 in self.phrase_id_dict:
                        ids_list.append(self.phrase_id_dict[phrase_3])
                        continue
                if i + 1 < len(words):
                    ids_list.append(self.phrase_id_dict[self.not_phrase_flag])
                else:
                    ids_list.append(self.phrase_id_dict[self.pad_flag])

            str_ids = " ".join([str(id) for id in ids_list])
            return str_ids
        else:
            return None

    def convert_emojis_ids(self, words):

        if len(words) >= 2:
            ids_list = [self.emoji_id_dict[self.unk_emoji_flag]]
            emoji_out_line = self.convert_words_to_emoji_combine_out(words)
            for emoji in emoji_out_line:
                if emoji in self.emoji_id_dict:
                    ids_list.append(self.emoji_id_dict[emoji])
                else:
                    ids_list.append(self.emoji_id_dict[self.unk_emoji_flag])

            str_ids = " ".join([str(id) for id in ids_list])
            return str_ids
        else:
            return None

    def convert_to_ids_file(self, data_path_out, is_train):
        phase = "train" if is_train else "dev"

        raw_file_reader = open(os.path.join(data_path_out, phase + "_data"), "r")
        words_id_file_writer = open(os.path.join(data_path_out, phase + "_in_ids_lm"), "w")
        letters_id_file_writer = open(os.path.join(data_path_out, phase + "_in_ids_letters"), "w")
        phrase_id_file_writer = open(os.path.join(data_path_out, phase + "_ids_phrase"), "w")
        emoji_id_file_writer = open(os.path.join(data_path_out, phase + "_ids_emoji"), "w")

        for line in raw_file_reader:
            line = line.rstrip()
            line_split = line.split("|#|")
            letters = line_split[0].split("\t")
            words = line_split[1].split("\t")

            if len(letters) != len(words):
                print(phase + " data line split error :", line)

            if len(words) >= 2:
                in_words_id = self.convert_in_words_ids(words)
                out_words_id = self.convert_out_words_ids(words)
                in_letters_id = self.convert_letters_ids(letters)
                phrase_id = self.convert_phrase_ids(words)
                emojis_id = self.convert_emojis_ids(words)

                if in_words_id and out_words_id and in_letters_id and phrase_id and emojis_id:
                    words_id_file_writer.write(in_words_id + "#" + out_words_id + "\n")
                    letters_id_file_writer.write(in_letters_id + "\n")
                    phrase_id_file_writer.write(phrase_id + "\n")
                    emoji_id_file_writer.write(emojis_id + "\n")

        raw_file_reader.close()
        words_id_file_writer.close()
        letters_id_file_writer.close()
        phrase_id_file_writer.close()
        emoji_id_file_writer.close()

        return

    def convert_to_ids(self, data_path_out):
        self.convert_to_ids_file(data_path_out, is_train=True)
        self.convert_to_ids_file(data_path_out, is_train=False)

        return


if __name__ == "__main__":

    args = sys.argv

    words_dict_file = args[1]
    emojis_file = args[2]
    emojis_combine_file = args[3]
    phrase_file = args[4]
    data_path_in = args[5]
    data_path_out = args[6]

    rate_threshold = float(args[7])
    words_num = int(args[8])
    phrase_num = int(args[9])
    train_data_num = int(args[10])
    dev_data_num = int(args[11])
    test_data_num = int(args[12])

    data_producer = TrainDataProducer()

    data_producer.emojis_dict = data_producer.load_vocab(emojis_file, "\t+", "emojis")
    data_producer.emojis_combine_dict = data_producer.load_vocab(emojis_combine_file, "\t+", "emojis combine")
    # data_producer.phrase_dict = data_producer.load_vocab(phrase_file, "##", "phrase")
    data_producer.full_words_dict = data_producer.load_vocab(words_dict_file, "\t+", "full")

    data_producer.calc_words_freq_all_data(data_path_in)
    data_producer.combine_words_and_phrase_dict(words_num, phrase_num)
    data_producer.convert_data(data_path_in, data_path_out, rate_threshold, train_data_num, dev_data_num, test_data_num)

    data_producer.save_vocab_files(data_path_out)
    data_producer.convert_to_ids(data_path_out)





