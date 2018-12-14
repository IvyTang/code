#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import re
from result import Result


class EfficiencyAnalyzer:

    def __init__(self, test_file, rnn_vocab, full_vocab, emoji_vocab, phrase_vocab):
        self.rnn_vocab = self.load_vocab(rnn_vocab, "##", "rnn")
        self.full_vocab = self.load_vocab(full_vocab, "\\t+", "full")
        self.emoji_vocab = self.load_vocab(emoji_vocab, "\\t+", "emoji")
        self.word_regex = "^[a-zA-Z'à-żÀ-ŻЀ-ӿḀ-ỹ]+$"
        self.phrase_vocab = self.load_vocab(phrase_vocab, "##", "phrase")

        result = Result()
        result.parse_resul_pc(test_file)
        self.res_sentences = result.res_sentences
        print("total " + str(len(self.res_sentences)) + " sentences")

        self.unk_flag = "<unk>"
        self.und_flag = "<und>"
        self.print_wrong_info = False
        self.wrong_input = dict()

        self.highlight_prob = 0.0

    def load_vocab(self, vocab_file, split_flag, vocab_type):
        vocab_dict = dict()
        with open(vocab_file, "r") as f:
            for line in f:
                line = line.strip()
                line = re.split(split_flag, line)
                if len(line) == 2:
                    token, id = line
                    vocab_dict[token] = int(id)
                else:
                    token = line[0]
                    vocab_dict[token] = 1
        f.close()
        print(vocab_type + " vocab num = " + str(len(vocab_dict)))
        return vocab_dict

    def is_equals(self, output_words, words_in):
        words = words_in.split()
        if len(output_words) < len(words) or len(words) == 0:
            return False
        for i in range(len(words)):
            out_word = output_words[i]
            if out_word not in self.rnn_vocab and out_word.lower() in self.rnn_vocab:
                out_word = out_word.lower()
            if words[i] != out_word:
                return False
        return True

    def is_contains(self, res_list, output_words):
        match_string = None
        for res in res_list:
            if self.is_equals(output_words, res):
                if match_string is None or len(res) > len(match_string):
                    match_string = res
        return match_string

    def end_of_emoji(self, input_context):
        if len(input_context) > 0:
            last_word = input_context[-1]
            if last_word in self.emoji_vocab:
                return True
        return False

    def is_emoji_contains(self, res_list, output_words):
        output_word = output_words[0]
        for res in res_list:
            if output_word in res:
                return True
        return False

    def is_letters_contains_emoji(self, input_letters):
        for letter in input_letters:
            if letter in self.emoji_vocab:
                return True
        return False

    def is_complete(self, input_letters, res_words):
        for res_word in res_words:
            if res_word.startswith(input_letters):
                return True
        return False

    def is_cap(self, input_letters, res_words):
        for res_word in res_words:
            if res_word.lower() == input_letters:
                return True
        return False

    def calc_unk_rate_one_line(self, words_line):
        unk_num = 0
        for word in words_line:
            if word not in self.rnn_vocab and word.lower() not in self.rnn_vocab:
                unk_num += 1
        return unk_num/len(words_line) if len(words_line) > 0 else 0.0

    def is_output_contains_phrase(self, output_words):
        if len(output_words) < 2:
            return False
        output_matched_phrase_2 = " ".join(output_words[:2])
        output_matched_phrase_3 = ""
        if len(output_words) >= 3:
            output_matched_phrase_3 = " ".join(output_words[:3])
        if output_matched_phrase_2 in self.phrase_vocab or output_matched_phrase_3 in self.phrase_vocab:
            return True
        return False

    def is_res_contains_phrase(self, res_list):
        for res in res_list:
            if len(res.split()) > 1:
                return True
        return False

    def cal_fraction(self, num, den):
        if den == 0:
            return "NaN"
        return str(num / den)

    def analyze(self, topn):
        sum_input_letter_num = 0.0
        sum_effective_letter_num = 0.0

        sum_word_input_letter_num = 0.0
        sum_word_effective_letter_num = 0.0

        sum_correct_num = 0.0
        sum_word_output_num = 0.0
        sum_emoji_output_num = 0.0
        sum_correct_word_num = 0.0
        sum_correct_emoji_num = 0.0

        sum_input_same_num = 0.0
        sum_input_same_wrong_num = 0.0

        is_highlight_num = 0.0
        is_highlight_wrong = 0.0
        is_highlight_complete_wrong = 0.0
        is_highlight_correct_wrong = 0.0
        is_highlight_complete_num = 0.0
        is_highlight_correct_num = 0.0

        sum_input_not_same_num = 0.0
        sum_input_not_same_wrong_num = 0.0

        sum_output_unk = 0.0
        sum_output_unk_regular = 0.0
        sum_output_und = 0.0
        sum_res_unk = 0.0
        sum_res_und = 0.0
        sum_correct_unk = 0.0
        sum_correct_und = 0.0
        sum_output_und_res_unk = 0.0

        sum_emoji = 0.0
        sum_word_to_emoji = 0.0
        sum_correct_word_to_emoji = 0.0
        sum_correct_word_to_emoji_combine = 0.0

        sum_output_word = 0.0
        sum_correct_lm_word = 0.0
        sum_res_not_emoji = 0.0

        sum_output_phrase = 0.0
        sum_correct_output_phrase = 0.0

        sum_phrase = 0.0
        sum_res_phrase = 0.0

        for res_sentence in self.res_sentences:
            res_words = res_sentence.res_words
            res_probs = res_sentence.res_probs
            input_letter_num = -1
            is_correct = False
            is_emoji = False
            is_phrase = False
            is_lm_correct = False
            is_unk = False
            is_und = False

            input_letters = "".join(res_sentence.input_letters)
            output_word = res_sentence.output_words[0]
            if output_word not in self.rnn_vocab and output_word.lower() in self.rnn_vocab:
                output_word = output_word.lower()
            match_word_final = ""

            for letter_num in range(len(res_words)):
                top_words = res_words[letter_num]
                top_probs = res_probs[letter_num]
                res_list = list()
                is_highlight = True
                for i in range(topn):
                    word = top_words[i]
                    prob = top_probs[i]
                    word_letter = "".join(res_sentence.input_letters[:letter_num])

                    if word in self.emoji_vocab:
                        is_emoji = True
                    if word in self.phrase_vocab:
                        is_phrase = True
                    if word == self.unk_flag or word == self.und_flag:
                        word = word_letter
                    if i==0 and prob <= self.highlight_prob and letter_num != 0:
                        word = word_letter
                        is_highlight = False

                    res_list.append(word)

                match_word_letter = self.is_contains(res_list, res_sentence.output_words)
                top1_res = top_words[0]

                top1_word = top1_res.split()[0]

                if letter_num == 0 and not self.end_of_emoji(res_sentence.input_context) \
                        and output_word in self.emoji_vocab:
                    if self.is_emoji_contains(res_list, res_sentence.output_words):
                        sum_correct_word_to_emoji_combine += 1

                if match_word_letter is not None:
                    if input_letter_num == -1:
                        input_letter_num = letter_num
                        match_word_final = match_word_letter
                    if letter_num == len(res_words) - 1:
                        is_correct = True
                    if letter_num == 0:
                        is_lm_correct = True
                        match_word_final_num = len(match_word_letter.split())
                        if top1_word == output_word:
                            if self.is_output_contains_phrase(res_sentence.output_words):
                                sum_output_phrase += 1
                                if match_word_final_num > 1:
                                    sum_correct_output_phrase += 1

                if letter_num != 0 and is_highlight:
                    is_highlight_num += 1
                    if match_word_letter is None:
                        is_highlight_wrong += 1
                    if word_letter != res_list[0]:
                        if self.is_complete(word_letter, res_list[0]):
                            is_highlight_complete_num += 1
                            if match_word_letter is None:
                                is_highlight_complete_wrong += 1
                        else:
                            is_highlight_correct_num += 1
                            if match_word_letter is None:
                                is_highlight_correct_wrong += 1

                if letter_num == 0:
                    if top1_word == output_word:
                        if self.is_res_contains_phrase(res_list):
                            sum_res_phrase += 1

                if letter_num == len(res_words) - 1 and output_word not in self.emoji_vocab:
                    if self.unk_flag in top_words[:topn]:
                        is_unk = True
                        sum_res_unk += 1
                    if self.und_flag in top_words[:topn]:
                        is_und = True
                        sum_res_und += 1

            if is_emoji:
                sum_emoji += 1
            if is_phrase:
                sum_phrase += 1

            effective_letter_num = len(match_word_final)
            if match_word_final in self.emoji_vocab:
                effective_letter_num = 1
            if input_letter_num == -1:
                input_letter_num = len(res_sentence.input_letters)
                effective_letter_num = 0

            sum_input_letter_num += input_letter_num
            sum_effective_letter_num += effective_letter_num

            if output_word in self.emoji_vocab:
                sum_emoji_output_num += 1
            else:
                sum_word_output_num += 1
                sum_word_input_letter_num += input_letter_num
                sum_word_effective_letter_num += effective_letter_num

            if input_letters == output_word and input_letters not in self.emoji_vocab:
                sum_input_same_num += 1
                if not is_correct:
                    sum_input_same_wrong_num += 1

            if input_letters != output_word and output_word not in self.emoji_vocab and len(input_letters) > 0 \
                    and not self.is_letters_contains_emoji(input_letters):
                sum_input_not_same_num += 1
                if not is_correct:
                    sum_input_not_same_wrong_num += 1

            if is_correct:
                sum_correct_num += 1
                if output_word in self.emoji_vocab:
                    sum_correct_emoji_num += 1
                else:
                    sum_correct_word_num += 1

            if output_word not in self.full_vocab:
                sum_output_unk += 1
                if re.match(self.word_regex, output_word):
                    sum_output_unk_regular += 1
                if is_unk:
                    sum_correct_unk += 1
            else:
                if output_word not in self.rnn_vocab:
                    sum_output_und += 1
                    if is_und:
                        sum_correct_und += 1
                    elif is_unk:
                        sum_output_und_res_unk += 1

            if not self.end_of_emoji(res_sentence.input_context) and output_word in self.emoji_vocab:
                sum_word_to_emoji += 1
                if is_correct:
                    sum_correct_word_to_emoji += 1

            if len(res_sentence.input_context) > 0 and output_word not in self.emoji_vocab:
                sum_output_word += 1
                if is_lm_correct:
                    sum_correct_lm_word += 1

                if not is_emoji:
                    sum_res_not_emoji += 1

        print("top " + str(topn) + " input efficiency = " + self.cal_fraction(sum_effective_letter_num, sum_input_letter_num) +
              ", accuracy = " + self.cal_fraction(sum_correct_num, len(self.res_sentences)))
        print("top " + str(topn) + " word input efficiency = " + self.cal_fraction(sum_word_effective_letter_num, sum_word_input_letter_num) +
              ", accuracy = " + self.cal_fraction(sum_correct_word_num, sum_word_output_num) + ", same wrong rate = " +
              self.cal_fraction(sum_input_same_wrong_num, sum_input_same_num) + ", not same wrong rate = " +
              self.cal_fraction(sum_input_not_same_wrong_num, sum_input_not_same_num))

        print("top " + str(topn) + " emoji popup rate = " + self.cal_fraction(sum_emoji, len(self.res_sentences)) + ", recall = " +
              self.cal_fraction(sum_correct_emoji_num, sum_emoji_output_num) + ", word to emoji recall = " +
              self.cal_fraction(sum_correct_word_to_emoji, sum_word_to_emoji) + ", word to emoji combine recall = " +
              self.cal_fraction(sum_correct_word_to_emoji_combine, sum_word_to_emoji))
        print("top " + str(topn) + " language model word recall = " + self.cal_fraction(sum_correct_lm_word, sum_output_word) +
              ", not emoji correct rate = " + self.cal_fraction(sum_res_not_emoji, sum_output_word))
        print("top " + str(topn) + " unk accuracy = " + self.cal_fraction(sum_correct_unk, sum_res_unk) + ", unk recall = " +
              self.cal_fraction(sum_correct_unk, sum_output_unk))
        print("top " + str(topn) + " und accuracy = " + self.cal_fraction(sum_correct_und, sum_res_und) + ", und recall = " +
              self.cal_fraction(sum_correct_und, sum_output_und) + ", und unk recall = " + self.cal_fraction(sum_output_und_res_unk, sum_output_und))
        print("top " + str(topn) + " phrase popup rate = " + self.cal_fraction(sum_phrase, len(self.res_sentences)) + " phrase recall = " +
              self.cal_fraction(sum_correct_output_phrase, sum_output_phrase) + " phrase acc = " + self.cal_fraction(sum_correct_output_phrase, sum_res_phrase))
        if topn==1:
            print("top 1 highlight wrong rate " + self.cal_fraction(is_highlight_wrong, is_highlight_num))


if __name__ == "__main__":

    args = sys.argv
    test_file = args[1]
    rnn_vocab = args[2]
    full_vocab = args[3]
    emoji_vocab = args[4]
    phrase_vocab = args[5]

    analyzer = EfficiencyAnalyzer(test_file, rnn_vocab, full_vocab, emoji_vocab, phrase_vocab)
    print()
    analyzer.analyze(1)
    print()
    analyzer.analyze(3)
