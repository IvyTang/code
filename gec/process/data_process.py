import spacy
from collections import namedtuple
import sys
import re
import numpy as np
import random
import os

Token = namedtuple('Token', ['text', 'lemma_', 'pos_', 'tag_'])


class Lemmatizer(object):
    def __init__(self, words_keys_pair_file, big_word_vocab, emoji_file):
        self._lemmatizer = spacy.load('en')
        self.max_line_count = -1
        self.pad_flag = "_PAD"
        self.unk_flag = "<unk>"
        self.bos_flag = "<bos>"
        self.eos_flag = "<eos>"
        self.num_str = "<num>"
        self.pun_str = "<pun>"
        self.start_flag = "<start>"
        self.emoji_flag = "<emoji>"
        self.word_collect = {}
        self.word_vocab = {}
        self.emoji_vocab = {}
        self.special_code_vocab = ["'ve", "'m", "'s", "n't", "'ll", "'re", "'t", "'d", "'cause", "'cos", "'em", "'im", "'round"]
        self.max_words_num = 50000
        self.train_rate = 0.99
        self.user_line = 0
        self.START_VOCABULART_WORD = [self.pad_flag, self.emoji_flag, self.unk_flag, self.bos_flag,
                                      self.eos_flag, self.num_str, self.pun_str]
        self.START_VOCABULART_WORD.extend(self.special_code_vocab)
        print(self.START_VOCABULART_WORD)
        self.START_VOCABULART_LETTER = [self.pad_flag, self.unk_flag, self.pun_str, self.start_flag]
        self.letter_collect = {}
        self.words_keys_pair = {}
        self.full_vocab = {}
        self.build_en_US_words_keys_pair(words_keys_pair_file)
        self.load_full_vocab(big_word_vocab)
        self.pun_list = [".", ",", ";", "?", "!", ":"]

        with open(emoji_file, mode="r") as f:
            for line in f:
                token, id = line.strip().split("\t")
                id = int(id)
                self.emoji_vocab[token] = id

        print("emoji vocabulary size =", str(len(self.emoji_vocab)))

    def is_word(self, word):
        if word in self.full_vocab or word in self.pun_list or word in self.special_code_vocab:
            return word
        elif word.lower() in self.full_vocab:
            return word.lower()
        return None

    def process_word(self, word):
        if word in self.full_vocab:
            return word
        elif word.lower() in self.full_vocab:
            return word.lower()
        return word

    def load_full_vocab(self, file):
        with open(file, "r") as f:
            for line in f:
                word, index = line.strip().split("\t")
                self.full_vocab[word] = index

    def post_process_tokens(self, tokens):
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
        
    def post_process_word(self, original_word, lemmatized_word):
        if lemmatized_word == '-PRON-':
            return original_word
        if original_word.isupper():
            return lemmatized_word.upper()
        if original_word.istitle():
            return lemmatized_word.capitalize()
        return lemmatized_word

    def get_lemmatized_idx(self, tokens):
        lemmatized_idx_array = []
        for i in range(len(tokens)):
            token = tokens[i]
            if token.pos_ in {'VERB', 'ADJ', 'ADV', 'NOUN'}:
                lemmatized_idx_array.append(int(i))
            else:
                if token.text.lower() != token.lemma_.lower() and token.lemma_ != '-PRON-':
                    print(token.text, token.lemma_, token.pos_, token.tag_)
        return lemmatized_idx_array

    def lemmatize(self, sentence):
        sentence = sentence.strip()
        tokens = self._lemmatizer(sentence)
        tokens_processed = self.post_process_tokens(tokens)
        words_original = [token.text for token in tokens_processed]
        words_lemmatized = [self.post_process_word(token.text, token.lemma_) for token in tokens_processed]
        lemmatized_idx_array = self.get_lemmatized_idx(tokens_processed)
        return words_original, words_lemmatized, lemmatized_idx_array

    def word_vocab_creater(self, words):
        for word in words:
            word_out = self.is_word(word)
            if word_out is not None and word_out not in self.special_code_vocab:
            # if word_out is not None:
                if word_out in self.word_collect:
                    self.word_collect[word_out] += 1
                else:
                    self.word_collect[word_out] = 1
                self.letter_vocab_creater(word_out)

    def create_token_file(self, file_in, file_out):
        with open(file_in, "r") as f_in,\
                 open(file_out, "w") as f_out:
            count = 0
            for line in f_in:
                count += 1
                if self.max_line_count > 0 and count >= self.max_line_count:
                    break
                if count > self.user_line:
                    sentence = line.strip().split("\t")[1]
                else:
                    sentence = line.strip().split("|#|")[1]
                    sentence = " ".join(sentence.split("\t"))
                if sentence != "":
                    if count > self.user_line:
                        sentence_original, sentence_lemmatized, lemmatized_idx = lemmatizer.lemmatize(sentence)
                    else:
                        sentence_original = sentence.split()
                        sentence_lemmatized = sentence.split()
                        lemmatized_idx = [0]
                    self.word_vocab_creater(sentence_original)

                    str_lemmatized_idx = [str(idx) for idx in lemmatized_idx]
                    f_out.write(" ".join(sentence_lemmatized) + "\t" + " ".join(sentence_original) + "\t"
                                + " ".join(str_lemmatized_idx) + "\n")

    def read_token_file(self, file_in):
        with open(file_in, "r") as f_in:
            for line in f_in:
                line_split = line.strip().split("\t")
                if len(line_split) == 3:
                    sentence_original = line_split[1].split()
                    self.word_vocab_creater(sentence_original)

    def gen_vocab(self, output_file_vocab_words, output_file_vocab_letters):

        words_vocab = self.START_VOCABULART_WORD + sorted(self.word_collect,
                                                          key=self.word_collect.get, reverse=True)
        letter_vocab = self.START_VOCABULART_LETTER + sorted(self.letter_collect, key=self.letter_collect.get,
                                                                    reverse=True)

        print("words vocab size:", len(words_vocab))
        print("letters vocab size:", len(letter_vocab))

        with open(output_file_vocab_words, "w") as f_words:
            self.word_vocab = self.vocab_file_write(words_vocab, f_words)

        with open(output_file_vocab_letters, "w") as f_letter:
            self.letter_vocab = self.vocab_file_write(letter_vocab, f_letter)

    def gen_vocab_freq(self, output_file_vocab_freq):
        word_freq_set = sorted(self.word_collect.items(), key=lambda x: x[1])

        print("word freq vocab size:", len(word_freq_set))

        with open(output_file_vocab_freq, "w") as f:
            for word_freq in word_freq_set:
                word, freq = word_freq
                f.write(word + "##" + str(freq) + "\n")
        f.close()

    def load_vocab(self, vocab_words, vocab_letters):

        with open(vocab_words, "r") as f_words:
            self.word_vocab = self.vocab_file_read(f_words)

        with open(vocab_letters, "r") as f_letter:
            self.letter_vocab = self.vocab_file_read(f_letter)

    def letter_vocab_creater(self, word):
        for letter in word:
            if self.is_letter(letter.lower()):
                if letter.lower() in self.letter_collect:
                    self.letter_collect[letter.lower()] += 1
                else:
                    self.letter_collect[letter.lower()] = 1

    def is_letter(self, letter):
        if re.match("[a-z]", letter) or letter == "'":
            return True
        return False

    def vocab_file_write(self, vocab_list, fp):
        vocab = {}
        count = 0
        for word in vocab_list:
            fp.write(word + "##" + str(count) + '\n')
            if word not in vocab:
                vocab[word] = count
            count += 1
        fp.close()
        return vocab

    def vocab_file_read(self, fp):
        vocab = {}
        for line in fp:
            word, id = line.strip().split("##")
            if word not in vocab: 
                vocab[word] = int(id)
            else:
                print(word,id)
        return vocab

    def train_ids_producer(self, file_in, data_path):
        train_file_in_letters = os.path.join(data_path, "train_in_ids_letters")
        train_file_in_lm = os.path.join(data_path, "train_in_ids_lm")

        dev_file_in_letters = os.path.join(data_path, "dev_in_ids_letters")
        dev_file_in_lm = os.path.join(data_path, "dev_in_ids_lm")

        with open(file_in, "r") as f_in, \
                open(train_file_in_letters, "w") as f_train_lm_out, \
                open(train_file_in_lm, "w") as f_train_letters_out, \
                open(dev_file_in_letters, "w") as f_dev_lm_out, \
                open(dev_file_in_lm, "w") as f_dev_letters_out:
            count = 0
            for line in f_in:
                count += 1
                line_split = line.strip().split("\t")
                if len(line_split) == 3:
                    sentence_lemmatized, sentence_original, lemmatized_idx = line_split
                    origin_words = sentence_original.split()
                    lemma_words = sentence_lemmatized.split()
                    if count > self.user_line:
                        letters = self.words_keys_pair_replace_random(origin_words)
                    else:
                        letters = self.words_keys_pair_replace(origin_words)
                    lemmatized_idx_split = lemmatized_idx.split()
                    words_num = len(origin_words)
                    if len(origin_words) != len(lemma_words):
                        print(origin_words, lemma_words)
                    elif int(lemmatized_idx_split[-1]) + 1 > len(origin_words):
                        print(origin_words, lemma_words, lemmatized_idx_split)
                    elif words_num >= 2:
                        train_prob = np.random.random_sample()
                        if train_prob < self.train_rate:
                            self.train_words_ids_producer(origin_words, lemma_words, lemmatized_idx, f_train_lm_out)
                            self.train_letter_ids_producer(letters, f_train_letters_out)
                        else:
                            self.train_words_ids_producer(origin_words, lemma_words, lemmatized_idx, f_dev_lm_out)
                            self.train_letter_ids_producer(letters, f_dev_letters_out)

    def train_words_ids_producer(self, origin_words, lemma_words, lemmatized_idx, file_out):
        origin_words_ids = [str(self.word_id_convert(word)) for word in origin_words]
        lemma_words_ids = [str(self.word_id_convert(word)) for word in lemma_words]
        file_out.write(" ".join(lemma_words_ids) +
                            "#" + str(self.word_vocab[self.bos_flag]) + " " + " ".join(origin_words_ids) +
                       " " + str(self.word_vocab[self.eos_flag]) +
                            "#" + lemmatized_idx + "\n")

    def word_id_convert(self, word):
        if word in self.word_vocab:
            word_id = self.word_vocab[word]
        elif word.lower() in self.word_vocab:
            word_id = self.word_vocab[word.lower()]
        elif word in self.emoji_vocab:
            word_id = self.word_vocab[self.emoji_flag]
        elif re.match("^[+-]*[0-9]+.*[0-9]*$", word):
            word_id = self.word_vocab[self.num_str]
        elif re.match("^[^a-zA-Z0-9']*$", word):
            word_id = self.word_vocab[self.pun_str]
        else:
            word_id = self.word_vocab[self.unk_flag]
        return word_id

    def letter_id_convert(self, letter):
        if letter in self.emoji_vocab:
            return ""
        if letter in self.letter_vocab:
            letter_id = self.letter_vocab[letter]
        elif letter in self.pun_list:
            letter_id = self.letter_vocab[self.pun_str]
        else:
            letter_id = self.letter_vocab[self.unk_flag]
        return letter_id

    def train_letter_ids_producer(self, words, file_out):
        letters_ids_str = []
        for word in words:
            # if word[0] != "'" and "'" in word:
            #     word = word.replace("'", "")
            letters_ids = [str(self.letter_id_convert(letter.lower())) for letter in word]
            letters_ids_str.append(" ".join(letters_ids))
        file_out.write("#".join(letters_ids_str) + "\n")

    def words_keys_pair_replace(self, words):
        letters = []
        for word in words:

            if word not in self.words_keys_pair and word.lower() not in self.words_keys_pair:
                letters.append(word.lower())
            else:
                if word in self.words_keys_pair:
                    replaced_word = word
                else:
                    replaced_word = word.lower()
                random_number_01 = np.random.random_sample()
                keys, scale = self.words_keys_pair[replaced_word]
                id = min([i for i in range(len(scale)) if scale[i] > random_number_01])
                letters.append(keys[id])
        return letters

    def words_keys_pair_replace_random(self, words):
        letters = []
        for word in words:
            random_number_01 = np.random.random_sample()
            if random_number_01 <= 0.333:
                letters.append(self.words_keys_pair_replace_edit1([word])[0])
            elif random_number_01 <= 0.666:
                letters.append(self.words_keys_pair_replace([word])[0])
            else:
                letters.append(self.words_keys_pair_replace_edit1([word], phase="real")[0])
        return letters

    def create_test_file(self, file_in, file_out):
        with open(file_in, "r") as f_in, \
                open(file_out, "w") as f_out:
            for line in f_in:
                line_split = line.strip().split("\t")
                if len(line_split) == 3:
                    _, sentence_original, _ = line_split
                    origin_words = sentence_original.split()
                    process_words = [self.process_word(word) for word in origin_words]
                    letters = self.words_keys_pair_replace(process_words)
                    words_num = len(process_words)
                    letters_num = len(letters)
                    if words_num != letters_num:
                        print(process_words, letters)
                    else:
                        letters = [" ".join(letter) for letter in letters]
                        f_out.write("\t".join(letters) + "|#|" + "\t".join(process_words) + "\n")

    def create_real_word_test_file(self, file_in, file_out):
        with open(file_in, "r") as f_in, \
                open(file_out, "w") as f_out:
            for line in f_in:
                line_split = line.strip().split("\t")
                if len(line_split) == 3:
                    _, sentence_original, _ = line_split
                    origin_words = sentence_original.split()
                    process_words = [self.process_word(word) for word in origin_words]
                    words = self.words_keys_pair_replace_edit1_real_word(process_words)
                    letters = [" ".join(word.lower()) for word in words]
                    f_out.write("\t".join(letters) + "|#|" + "\t".join(process_words) + "\n")

    def create_user_test_file(self, file_in, file_out):
        with open(file_in, "r") as f_in, \
                open(file_out, "w") as f_out:
            for line in f_in:
                line_split = line.rstrip().split("|#|")
                if len(line_split) == 2:
                    keys_line, words_line = line_split
                    keys = keys_line.split("\t")
                    words = words_line.split("\t")
                    new_keys = []
                    for (key, word) in zip(keys, words):
                        if key == "":
                            new_keys.append(key)
                        else:
                            # new_keys.append(self.words_keys_pair_replace([word])[0])
                            new_keys.append(self.words_keys_pair_replace_edit1([word])[0])
                    print(new_keys)
                    letters = [" ".join(letter) for letter in new_keys]
                    f_out.write("\t".join(letters) + "|#|" + "\t".join(words) + "\n")

    def build_en_US_words_keys_pair(self, file):
        with open(file, "r") as f:
            for line in f:
                word, key, freq = line.strip().split("\t")
                if word not in self.words_keys_pair:
                    self.words_keys_pair[word] = {}
                self.words_keys_pair[word][key] = int(freq)
        # has_wrong_word_count = 0
        for word in self.words_keys_pair.keys():
            has_correct_word = False
            keys_dict = self.words_keys_pair[word]
            freq_list = list(keys_dict.values())
            for key in list(keys_dict.keys()):
                if key == word.lower():
                    has_correct_word = True
                    break
            new_key_set = list(keys_dict.keys())
            if not has_correct_word and " " not in word and "'" not in word:
                new_key_set.insert(0, word)
                first_freq = freq_list[0]
                freq_list.insert(0, 3 * first_freq)
                # print(word, list(keys_dict.keys()))
                # word_original, word_lemmatized, lemmatized_idx = self.lemmatize(word)
                # print(word_original, word_lemmatized, lemmatized_idx)
                # has_wrong_word_count += 1
                # print(new_key_set, word, freq_list)
            elif has_correct_word and " " not in word and "'" not in word:
                first_key = new_key_set[0]
                if first_key != word.lower():
                    word_index = new_key_set.index(word.lower())
                    new_key_set[0] = word.lower()
                    new_key_set[word_index] = first_key
                    first_freq = freq_list[0]
                    freq_list[0] = 3 * first_freq
                    # print(new_key_set, word, freq_list)

                    # has_wrong_word_count += 1

            num_samples = float(sum(freq_list))
            num_scale = [sum(freq_list[:i + 1]) / num_samples for i in range(len(freq_list))]
            self.words_keys_pair[word] = (new_key_set, num_scale)
        # print(has_wrong_word_count)

    def edits1(self, word):
        """与'word'的编辑距离为1的全部结果（不一定在词表里）"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits for c in letters if len(R) > 0]
        inserts = [L + c + R for L, R in splits for c in letters]

        return [deletes, transposes, replaces, inserts]

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
        deletes = [word for word in deletes if word in self.word_vocab and word != correct_word]

        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        transposes = [word for word in transposes if word in self.word_vocab and word != correct_word]

        replaces = [L + c + R[1:] for L, R in splits for c in letters if len(R) > 0]
        replaces = [word for word in replaces if word in self.word_vocab and word != correct_word]

        inserts = [L + c + R for L, R in splits for c in letters]
        inserts = [word for word in inserts if word in self.word_vocab and word != correct_word]

        real_word_edit1_lists = [deletes, transposes, replaces, inserts]
        while [] in real_word_edit1_lists:
            real_word_edit1_lists.remove([])

        if len(real_word_edit1_lists) == 0:
            return None
        else:
            return self.flatten(real_word_edit1_lists)

    def words_keys_pair_replace_edit1(self, words, phase="unk"):
        letters = []
        non_words_list = None
        real_words_list = None

        for word in words:
            if "'" in word or " " in word or len(word) == 1 or \
                    (word not in self.words_keys_pair and word.lower() not in self.words_keys_pair):
                letters.append(self.words_keys_pair_replace([word])[0])
            else:
                random_number_01 = np.random.random_sample()
                if random_number_01 <= 0.5:
                    if phase == "real":
                        real_words_list = self.real_word_edits1(word.lower())
                        if real_words_list is None:
                            non_words_list = self.edits1(word.lower())
                    else:
                        non_words_list = self.edits1(word.lower())
                    if non_words_list is not None:
                        wrong_words_list_one_type = random.sample(non_words_list, 1)[0]
                        wrong_words = random.sample(wrong_words_list_one_type, 1)[0]
                    else:
                        wrong_words = random.sample(real_words_list, 1)[0]
                    letters.append(wrong_words)
                else:
                    letters.append(word.lower())
        return letters

    def words_keys_pair_replace_edit1_real_word(self, words):
        raw_words = [word for word in words]
        real_words_error_list = []
        for word in raw_words:
            if "'" in word or " " in word or len(word) == 1 or \
                    (word not in self.words_keys_pair and word.lower() not in self.words_keys_pair):
                real_words_error_list.append(word.lower())
            else:
                real_words_list = self.real_word_edits1(word.lower())
                if real_words_list is not None:
                    wrong_words = random.sample(real_words_list, 1)[0]
                else:
                    wrong_words = word.lower()
                real_words_error_list.append(wrong_words)
        if real_words_error_list == [word.lower() for word in raw_words]:
            return raw_words
        else:
            while True:
                index = random.sample([i for i in range(len(real_words_error_list))], 1)[0]
                if real_words_error_list[index] != raw_words[index].lower():
                    raw_words[index] = real_words_error_list[index]
                    break

            return raw_words

    def word_only_id_produce(self, file_words_out, file_letters_out, copy_num):
        with open(file_words_out, "w") as f_words_out, \
                open(file_letters_out, "w") as f_letters_out:
            for i in range(copy_num):
                for word in self.word_vocab:
                    if word not in [self.pad_flag, self.unk_flag, self.bos_flag, self.eos_flag,
                                    self.num_str, self.pun_str, self.start_flag, self.emoji_flag]:

                        lemmatized_idx = "0"
                        self.train_words_ids_producer([word], [word], lemmatized_idx, f_words_out)
                        self.train_letter_ids_producer([word], f_letters_out)

    def create_lemma_vocab(self, file_words_out, file_lemma_words_out):
        with open(file_words_out, "r") as f_words_out, \
                open(file_lemma_words_out, "w") as f_lemma_words_out:
            lemma_vocab = {}
            for line in f_words_out:
                word, id = line.strip().split("##")
                if word in self.START_VOCABULART_WORD:
                    continue
                word_original, word_lemmatized, lemmatized_idx = lemmatizer.lemmatize(word)
                if len(lemmatized_idx) > 0 and len(word_original) == 1 and len(word_lemmatized) == 1:
                    word_original = word_original[0]
                    word_lemmatized = word_lemmatized[0]
                    if word_lemmatized not in lemma_vocab:
                        lemma_vocab[word_lemmatized] = []
                    if word_original not in lemma_vocab[word_lemmatized]:
                        lemma_vocab[word_lemmatized].append(word_original)
            for lemma_word in lemma_vocab:
                words_set = lemma_vocab[lemma_word]
                if len(words_set) > 1:
                    f_lemma_words_out.write("\t".join(words_set) + "\n")


if __name__ == "__main__":

    args = sys.argv

    lemmatizer = Lemmatizer("vocab/word_keys_pair_merged.txt", "vocab/main_en_1_unigram", "vocab/emojis")
    lemmatizer.create_token_file("../resource/lang8_sample.txt", "../resource/lang8_lemma_sample.txt")
    lemmatizer.gen_vocab("../resource/train_data/vocab_in_words",
                         "../resource/train_data/vocab_in_letters")
    # lemmatizer.load_vocab("../resource/train_data/vocab_in_words", "../resource/train_data/vocab_in_letters")
    lemmatizer.train_ids_producer("../resource/lang8_lemma_sample.txt", "../resource/train_data/")
