import tensorflow as tf
import os
import time

from data_utility import DataUtility
from seq2word_word_letter_sparse_with_bucket import Config

class InputEngineSparse(object):
    def __init__(self, model_path, config_name):

        vocab_file_in_words = os.path.join(model_path, "vocab_in_words")
        vocab_file_in_letters = os.path.join(model_path, "vocab_in_letters")
        vocab_file_out = os.path.join(model_path, "vocab_out")
        config_file = os.path.join(model_path, config_name)

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

        saved_model_path = os.path.join(model_path, 'sparse_graph-finetune-' + config_name + '.pb')
        with open(saved_model_path, 'rb') as f:
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
        return [{'word': word, 'probability': float(probability)}
                if word != '<unk>' else {'word': '<' + word_letters + '>', 'probability': float(probability)}
                for word, probability in zip(words_out, probability_topk)] if len(words_out) > 0 else []

    def predict_data(self, sentence):
        sentence = sentence.rstrip()
        inputs, words_num, letters_num = self._data_utility.data2ids_line(sentence)
        if inputs == None:
            return None
        words_out = []
        state_out = None
        for i in range(len(inputs)):
            feed_values = {self.input_name: [[inputs[i]]], self.top_k_name: 3}
            if i > 0:
                feed_values[self.state_in_name] = state_out
            probabilities, top_k_predictions, state_out = self._sess.run([self.output_name, self.top_k_prediction_name,
                                                                          self.state_out_name], feed_dict=feed_values)
            words = self._data_utility.ids2outwords(top_k_predictions[0])
            words_out.append(words)
        out_str = str(words_out[words_num - 1 : words_num + letters_num] if words_num > 0 else [['', '', '']] + words_out[0 : letters_num])
        return out_str

    def predict_file(self, test_file_in, test_file_out):
        testfilein = open(test_file_in, "r")
        testfileout = open(test_file_out, 'w')
        t1 = time.time()
        for sentence in testfilein:
            sentence = sentence.rstrip()
            out_str = self.predict_data(sentence)
            if (out_str):
                print (sentence + " |#| " + out_str)
                testfileout.write(sentence + " |#| " + out_str + "\n")
            else:
                print ("predict error : " + sentence)
        t2 = time.time()
        print(t2 - t1)
        testfilein.close()
        testfileout.close()

    def predict_data_probability(self, sentence):
        sentence = sentence.rstrip()
        inputs, words_num, letters_num = self._data_utility.data2ids_line(sentence)
        if inputs == None:
            return None
        words_out = []
        probability_out = []
        state_out = None
        for i in range(len(inputs)):
            feed_values = {self.input_name: [[inputs[i]]], self.top_k_name: 3}
            if i > 0:
                feed_values[self.state_in_name] = state_out
            probabilities, top_k_predictions, state_out = self._sess.run([self.output_name, self.top_k_prediction_name,
                                                                          self.state_out_name], feed_dict=feed_values)
            top3 = top_k_predictions[0]
            probability_top3 = [probabilities[0][id] for id in top3]
            words = self._data_utility.ids2outwords(top3)
            words_out.append(words)
            probability_out.append(probability_top3)

        out_str = ''
        if words_num > 0:
            words_out_use = words_out[words_num - 1: words_num + letters_num]
            probability_out_use = probability_out[words_num - 1: words_num + letters_num]
            for words, probabilities in zip(words_out_use, probability_out_use):
                out_str_line = ''
                for word,probability in zip(words, probabilities):
                    out_str_line = out_str_line + " | " + word + ' # ' + '{:.8f}'.format(probability)
                out_str_line = out_str_line[3:-1]
                out_str = out_str + " || " + out_str_line
            out_str = out_str[4:-1]
        else:
            words_out_use = words_out[0: letters_num]
            probability_out_use = probability_out[0: letters_num]
            for words, probabilities in zip(words_out_use, probability_out_use):
                out_str_line = ''
                for word, probability in zip(words, probabilities):
                    out_str_line = out_str_line + " | " + word + ' # ' + '{:.8f}'.format(probability)
                out_str_line = out_str_line[3:-1]
                out_str = out_str + " || " + out_str_line
        return out_str

    def predict_file_probability(self, test_file_in, test_file_out):
        testfilein = open(test_file_in, "r")
        testfileout = open(test_file_out, 'w')
        t1 = time.time()
        for sentence in testfilein:
            sentence = sentence.rstrip()
            out_str = self.predict_data_probability(sentence)
            if (out_str):
                print (sentence + " |#| " + out_str)
                testfileout.write(sentence + " |#| " + out_str + "\n")
            else:
                print ("predict error : " + sentence)
        t2 = time.time()
        print(t2 - t1)
        testfilein.close()
        testfileout.close()

if __name__ == "__main__":
    model_path = "/home/gaoxin/workspace/python/dl-tensorflow-dev/seq2word_word_letter_sparse/resource/model/201612_201708_cleaned_20000_08_6000W_out2unk_bucket_sparse/"
    config_name = "20032_20002_400_2000_10_bucket.cfg"
    engine = InputEngineSparse(model_path, config_name)

    test_file_in = "resource/test_data/test_data_cleaned_50031"
    test_file_out = "target/201612_201708_cleaned_20000_08_6000W_out2unk_bucket_sparse_6_cleaned_50031"
    engine.predict_file(test_file_in, test_file_out)

    # test_file_in = "resource/test_data/test_data_50189"
    # test_file_out = "target/2016_2017_20000_08_2000W_ids_combine_fix_fixNoWord_400_sparse_2_991328_1_50189_probability"
    # engine.predict_file_probability(test_file_in, test_file_out)

    # sentence = "no I cant"
    # res = engine.predict(sentence, 10)
    # print(sentence)
    # print(res)