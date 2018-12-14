import tensorflow as tf
import os
import time
import sys

from data_utility import DataUtility
from seq2word_word_letter_sparse_with_bucket import Config


class InputEngineSparse(object):
    def __init__(self, model_file, vocab_path, config_file):

        vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")
        vocab_file_in_letters = os.path.join(vocab_path, "vocab_in_letters")
        vocab_file_out = os.path.join(vocab_path, "vocab_out")

        self.max_test_line = 10000
        config = Config()
        config.get_config(config_file)
        self._data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words,
                                         vocab_file_in_letters=vocab_file_in_letters,
                                         vocab_file_out=vocab_file_out, max_sentence_length=config.num_steps)

        self.sparsity = config.sparsity
        prefix = "import/"
        self.top_k_name = prefix + "Online/Model/top_k:0"
        self.state_in_name = prefix + "Online/Model/state:0"
        self.input_name = prefix + "Online/Model/batched_input_word_ids:0"

        self.top_k_prediction_name = prefix + "Online/Model/top_k_prediction:1"
        self.output_name = prefix + "Online/Model/probabilities:0"
        self.state_out_name = prefix + "Online/Model/state_out:0"

        # saved_model_path = os.path.join(model_path, 'sparse_graph-finetune-' + config_name + '.pb')
        with open(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_fraction
        self._sess = tf.Session(config=gpu_config)

    def predict(self, sentence, k):
        """Feed a sentence (str) and perform inference on this sentence """
        global probabilities, top_k_predictions

        sentence_ids, word_letters = self._data_utility.sentence2ids(sentence)
        if len(sentence_ids) == 0:
            return ''

        # Feed input sentence word by word.
        state_out = None
        for i in range(len(sentence_ids)):
            feed_values = {self.input_name: [[sentence_ids[i]]],
                           self.top_k_name: k}
            if i > 0:
                feed_values[self.state_in_name] = state_out
            # probabilities is an ndarray of shape (batch_size * time_step) * vocab_size
            # For inference, batch_size = num_step = 1, thus probabilities.shape = 1 * vocab_size
            probabilities, top_k_predictions, state_out = self._sess.run([self.output_name, self.top_k_prediction_name,
                                                                          self.state_out_name], feed_dict=feed_values)

        probability_topk = [probabilities[0][id] for id in top_k_predictions[0]]
        words_out = self._data_utility.ids2outwords(top_k_predictions[0])
        str_out = ''
        for word, probability in zip(words_out, probability_topk):
            str_out = str_out + '{\'word\': \'' + word + '\', \'probability\': ' + str(probability) + '}, '
        str_out = '[' + str_out[0: len(str_out) - 2] + ']'
        return str_out
        # return [{'word': word, 'probability': float(probability)}
        #         if word != '<unk>' else {'word': '<' + word_letters + '>', 'probability': float(probability)}
        #         for word, probability in zip(words_out, probability_topk)] if len(words_out) > 0 else []

    def predict_data(self, sentence, k):
        sentence = sentence.rstrip()
        inputs, words_num, letters_num = self._data_utility.data2ids_line(sentence)
        if inputs == None:
            return None
        words_out = []
        state_out = None
        for i in range(len(inputs)):
            feed_values = {self.input_name: [[inputs[i]]], self.top_k_name: k}
            if i > 0:
                feed_values[self.state_in_name] = state_out
            probabilities, top_k_predictions, state_out = self._sess.run([self.output_name, self.top_k_prediction_name,
                                                                          self.state_out_name], feed_dict=feed_values)
            words = self._data_utility.ids2outwords(top_k_predictions[0])
            words_out.append(words)
        res_list_all = words_out[words_num - 1: words_num + letters_num] if words_num > 0 else [['', '', '']] + words_out[0: letters_num]
        res_str_all = ''
        for res_list_one in res_list_all:
            res_str_one = ''
            for word in res_list_one:
                res_str_one = res_str_one + '\'' + word + '\', '
            res_str_one = res_str_one[0: len(res_str_one) - 2]
            res_str_all = res_str_all + '[' + res_str_one + '], '
        res_str_all = '[' + res_str_all[0: len(res_str_all) - 2] + ']'
        return res_str_all
        # out_str = str(words_out[words_num - 1 : words_num + letters_num] if words_num > 0 else [['', '', '']] + words_out[0 : letters_num])
        # return out_str

    def predict_file(self, test_file_in, test_file_out, k):
        testfilein = open(test_file_in, "r", encoding='utf-8')
        testfileout = open(test_file_out, 'w', encoding='utf-8')
        t1 = time.time()
        for sentence in testfilein:
            sentence = sentence.rstrip()
            out_str = self.predict_data(sentence, k)
            if (out_str):
                print (sentence + " |#| " + out_str)
                testfileout.write(sentence + " |#| " + out_str + "\n")
            else:
                print ("predict error : " + sentence)
        t2 = time.time()
        print(t2 - t1)
        testfilein.close()
        testfileout.close()

    def predict_data_probability(self, sentence, k):
        sentence = sentence.rstrip()
        inputs, words_num, letters_num = self._data_utility.data2ids_line(sentence)
        if inputs == None:
            return None
        words_out = []
        probability_out = []
        state_out = None
        for i in range(len(inputs)):
            feed_values = {self.input_name: [[inputs[i]]], self.top_k_name: k}
            if i > 0:
                feed_values[self.state_in_name] = state_out
            probabilities, top_k_predictions, state_out = self._sess.run([self.output_name, self.top_k_prediction_name,
                                                                          self.state_out_name], feed_dict=feed_values)
            topk = top_k_predictions[0]
            probability_topk = [probabilities[0][id] for id in topk]
            words = self._data_utility.ids2outwords(topk)
            words_out.append(words)
            probability_out.append(probability_topk)

        out_str = ''
        if words_num > 0:
            words_out_use = words_out[words_num - 1: words_num + letters_num]
            probability_out_use = probability_out[words_num - 1: words_num + letters_num]
            for words, probabilities in zip(words_out_use, probability_out_use):
                out_str_line = ''
                for word,probability in zip(words, probabilities):
                    out_str_line = out_str_line + ", '" + word + ':' + '{:.8f}'.format(probability) + "'"
                out_str_line = out_str_line[2:len(out_str_line)]
                out_str_line = "[" + out_str_line + "]"
                out_str = out_str + ", " + out_str_line
            out_str = "[" + out_str[2:len(out_str)] + "]"
        else:
            words_out_use = words_out[0: letters_num]
            probability_out_use = probability_out[0: letters_num]
            for words, probabilities in zip(words_out_use, probability_out_use):
                out_str_line = ''
                for word, probability in zip(words, probabilities):
                    out_str_line = out_str_line + ", '" + word + ':' + '{:.8f}'.format(probability) + "'"
                out_str_line = out_str_line[2:len(out_str_line)]
                out_str_line = "[" + out_str_line + "]"
                out_str = out_str + ", " + out_str_line
            out_str = "[['', '', ''], " + out_str[2:len(out_str)] + "]"
        return out_str

    def predict_file_probability(self, test_file_in, test_file_out, k):
        testfilein = open(test_file_in, "r", encoding='utf-8')
        testfileout = open(test_file_out, 'w', encoding='utf-8')
        t1 = time.time()
        for sentence in testfilein:
            sentence = sentence.rstrip()
            out_str = self.predict_data_probability(sentence, k)
            if (out_str):
                print (sentence + " |#| " + out_str)
                testfileout.write(sentence + " |#| " + out_str + "\n")
            else:
                print ("predict error : " + sentence)
        t2 = time.time()
        print(t2 - t1)
        testfilein.close()
        testfileout.close()

    def contains_emoji(self, input_letters):
        input_letters_array = input_letters.split('\t')
        return '' in input_letters_array

    def predict_ids(self, inputs, words_num, k):
        words_out = []
        probability_out = []
        state_out = None
        for i in range(len(inputs)):
            feed_values = {self.input_name: [[inputs[i]]], self.top_k_name: k}
            if i > 0:
                feed_values[self.state_in_name] = state_out
            probabilities, top_k_predictions, state_out = self._sess.run([self.output_name, self.top_k_prediction_name,
                                                                          self.state_out_name], feed_dict=feed_values)
            topk = top_k_predictions[0]
            probability_topk = [probabilities[0][id] for id in topk]
            words = self._data_utility.ids2outwords(topk)
            words_out.append(words)
            probability_out.append(probability_topk)
        if words_num > 0:
            words_out = words_out[words_num - 1:]
            probability_out = probability_out[words_num - 1:]
        else:
            words_out = [["" for _ in range(k)]] + words_out
            probability_out = [[0.0 for _ in range(k)]] + probability_out
        return words_out, probability_out

    def predict_data_new(self, line, k):
        input_letters_sentence = ''
        line_array = line.split('|#|')
        input_letters = line_array[0]
        output_words = line_array[1]
        topk_list = []
        probability_list = []

        # if not self.contains_emoji(input_letters):
        input_letters_array = input_letters.split('\t')
        output_words_array = output_words.split('\t')
        if len(input_letters_array) == len(output_words_array):
            output_words_ids_array = self._data_utility.words2ids(' '.join(output_words_array))
            for i in range(len(input_letters_array)):
                word = input_letters_array[i].replace(' ', '')
                input_letters_sentence = input_letters_sentence + word + ' '
                inputs = output_words_ids_array[0:i]
                words_num = len(inputs)
                letter_ids = self._data_utility.letters2ids(" ".join(input_letters_array[i]))
                inputs.extend(letter_ids)
                topk, probability = self.predict_ids(inputs, words_num, k)
                topk_list.append(topk)
                probability_list.append(probability)

            return output_words_array, input_letters_array, topk_list, probability_list
        else:
            return None

    def result_print(self, out_string, out_prob):
        string = ""
        for (word, prob) in zip(out_string, out_prob):
            prob = str(prob) if word != "" else "0.0"
            string = string + word + ":" + prob + "|"
        string = string[:-1]
        return string

    def predict_file_new(self, infile, outfile):
        t1 = time.time()
        line_count = 0

        with open(infile, mode="r", encoding='utf-8') as fi:
            with open(outfile, mode="w", encoding='utf-8') as fo:
                for data_line in fi:
                    line_count += 1
                    if line_count > self.max_test_line:
                        break

                    data_line = data_line.rstrip()
                    res = self.predict_data_new(data_line, 3)
                    if res:
                        words_line, letters_line, out_words_list, out_prob_list = res

                        for i in range(len(out_words_list)):
                            # print("\t".join(words_line[:i])
                            #       + "|#|" + " ".join(letters_line[i])
                            #       + "|#|" + "\t".join(words_line[i:]) + "|#|"
                            #       + '\t'.join([self.result_print(out_words, out_prob)
                            #                    for (out_words, out_prob) in zip(out_words_list[i], out_prob_list[i])])
                            #       + "\n")
                            fo.write("\t".join(words_line[:i])
                                              + "|#|" + " ".join(letters_line[i])
                                              + "|#|" + "\t".join(words_line[i:]) + "|#|"
                                              + '\t'.join([self.result_print(out_words, out_prob)
                                                           for (out_words, out_prob) in
                                                           zip(out_words_list[i], out_prob_list[i])])
                                              + "\n")
        t2 = time.time()
        print(t2 - t1)
        fi.close()
        fo.close()


if __name__ == "__main__":
    args = sys.argv

    model_file = args[1]
    vocab_path = args[2]
    config_file = args[3]
    test_file_in = args[4]
    test_file_out = args[5]

    engine = InputEngineSparse(model_file, vocab_path, config_file)
    engine.predict_file_new(test_file_in, test_file_out)
