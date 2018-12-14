import re

class DataUtility:
    def __init__(self, vocab_file_in_words=None, vocab_file_in_letters=None, vocab_file_out=None, corpus_file=None, max_sentence_length=30):
        self.unkw_str = "<unk>"
        self.unkl_str = "<unk>"
        self.unk_str = "<unk>"
        self.num_str = "<num>"
        self.pun_str = "<pun>"
        self.pad_id = 0
        self.max_sentence_length = max_sentence_length
        if (vocab_file_in_words and vocab_file_in_letters and vocab_file_out):
            self.id2token_in, self.id2token_out = {}, {}
            self.token2id_in_words, self.token2id_in_letters, self.token2id_out = {}, {}, {}
            with open(vocab_file_in_words, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in[id] = token
                    self.token2id_in_words[token] = id
            print("in words vocabulary size =", str(len(self.token2id_in_words)))
            with open(vocab_file_in_letters, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in[id] = token
                    self.token2id_in_letters[token] = id
            print("in letters vocabulary size =", str(len(self.token2id_in_letters)))
            print("in vocabulary size =", str(len(self.id2token_in)))
            with open(vocab_file_out, mode="r") as f:
                for line in f:
                    token, id = line.split("##")
                    id = int(id)
                    self.id2token_out[id] = token
                    self.token2id_out[token] = id
            print("out vocabulary size =", str(len(self.token2id_out)))

    def word2id(self, word):
        if word in self.token2id_in_words:
            word_out = word
        elif word.lower() in self.token2id_in_words:
            word_out = word.lower()
        elif re.match("^[+-]*[0-9]+.*[0-9]*$", word):
            word_out = self.num_str
        elif re.match("^[^a-zA-Z0-9']*$", word):
            word_out = self.pun_str
        else:
            word_out = self.unkw_str
        return self.token2id_in_words.get(word_out, self.token2id_in_words[self.unkw_str])

    def words2ids(self, words):
        words_split = re.split("\\s+", words)
        return [self.word2id(word) for word in words_split if len(word) > 0]

    def letters2ids(self, letters):
        letters_split = re.split("\\s+", letters)
        return [self.token2id_in_letters.get(letter.lower(), self.token2id_in_letters[self.unkl_str]) for letter in letters_split if len(letter) > 0]

    def outword2id(self, outword):
        return self.token2id_out.get(outword, self.token2id_out[self.unk_str])

    def ids2outwords(self, ids_out):
        return [self.id2token_out.get(id, self.unk_str) for id in ids_out]

    def data2ids_line(self, data_line):
        data_line_split = re.split("\\t+", data_line)
        words = data_line_split[0].strip()
        letters = data_line_split[1].strip()
        letters = letters[3 : len(letters) - 4].strip()
        words_ids = self.words2ids(words)
        letters_ids = self.letters2ids(letters)
        # outword = data_line_split[2].strip()
        # out_id = self.outword2id(outword)
        words_num = len(words_ids)
        letters_num = len(letters_ids)
        input = words_ids + letters_ids
        return input, words_num, letters_num

    def sentence2ids(self, sentence):
        words_array = re.split('\\s+', sentence)
        word_letters = words_array[-1]
        words_array = words_array[:-1]
        words = ' '.join(words_array)
        letters = ' '.join(word_letters)
        words_ids = self.words2ids(words)
        letters_ids = self.letters2ids(letters)
        return words_ids + letters_ids, word_letters

    def data2ids_line_pad(self, data_line):
        data_line_split = re.split("\\t+", data_line)
        words = data_line_split[0].strip()
        letters = data_line_split[1].strip()
        letters = letters[3 : len(letters) - 4].strip()
        outword = data_line_split[2].strip()
        words_ids = self.words2ids(words)
        letters_ids = self.letters2ids(letters)
        out_id = self.outword2id(outword)
        words_num = len(words_ids)
        letters_num = len(letters_ids)

        input = None
        if letters_num <= self.max_sentence_length:
            if words_num + letters_num <= self.max_sentence_length:
                input = words_ids + letters_ids + [self.pad_id] * (self.max_sentence_length - words_num - letters_num)
            else:
                if letters_num < self.max_sentence_length:
                    input = words_ids[-(self.max_sentence_length - letters_num):] + letters_ids
                else:
                    input = letters_ids
                words_num = self.max_sentence_length - letters_num

        output = [self.pad_id] * max(words_num - 1, 0) + \
                 [out_id] * (letters_num + 1 if words_num > 0 else letters_num) + \
                 [self.pad_id] * (self.max_sentence_length - words_num - letters_num)
        mask = [0.0] * max(words_num - 1, 0) + \
               [1.0] * (letters_num + 1 if words_num > 0 else letters_num) + \
               [0.0] * (self.max_sentence_length - words_num - letters_num)

        return input, output, mask, words_num, letters_num


if __name__ == "__main__":
    du = DataUtility(vocab_file_in_words="resource/model/2016_2017_20000_08_2000W_ids_combine_400/vocab_in_words",
                     vocab_file_in_letters="resource/model/2016_2017_20000_08_2000W_ids_combine_400/vocab_in_letters",
                     vocab_file_out="resource/model/2016_2017_20000_08_2000W_ids_combine_400/vocab_out")

    data_line = "	<b> w y d </b>		wyd"
    input, _, _ = du.data2ids_line(data_line)
    print(input)

    data_line = "happy new year 34 yadaseqwasd	<b> d s </b>		ds"
    input, _, _ = du.data2ids_line(data_line)
    print(input)

    input, output, mask, words_num, letters_num = du.data2ids_line_pad(data_line)
    print (input)
    print (output)
    print (mask)
    outwords = du.ids2outwords(output)
    print (outwords)
    print (words_num, letters_num)

    sentence = "happy new year 34 yadaseqwasd ds"
    input, word_letters = du.sentence2ids(sentence)
    print (input)
