from __future__ import print_function

import os
import sys
import time

from data_utility import DataUtility
from seq2word_word_letter_sparse_with_bucket import PTBModel, Config
# from seq2word_word_letter_sparse import PTBModel, Config
import tensorflow as tf

class InputEngineRnn:

    def __init__(self, model_path, model_name, config_name, 
                 full_vocab_path=None):
        vocab_file_in_words = os.path.join(model_path, "vocab_in_words")
        vocab_file_in_letters = os.path.join(model_path, "vocab_in_letters")
        vocab_file_out = os.path.join(model_path, "vocab_out")
        model_file = os.path.join(model_path, model_name)
        config_file = os.path.join(model_path, config_name)

        self._config = Config()
        self._config.get_config(config_file)
        self._data_utility = DataUtility(
            vocab_file_in_words=vocab_file_in_words, 
            vocab_file_in_letters=vocab_file_in_letters,
            vocab_file_out=vocab_file_out, 
            max_sentence_length=self._config.num_steps,
            full_vocab_file_in_words=full_vocab_path)
        self._config.batch_size = 1
        self._config.num_steps = 1

        with tf.Graph().as_default():
            with tf.variable_scope("Model"):
                self._language_model_test = PTBModel(is_training=False, config=self._config, bucket=1)

            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = self._config.gpu_fraction
            self._sess = tf.Session(config=gpu_config)
            with self._sess.as_default():
                # Do not restore sparse weights from pretrain phase
                restore_variables = dict()
                for v in tf.trainable_variables():
                    if v.name.startswith("Model/Softmax/softmax_sp_trainable_weights") \
                            or v.name.startswith("Model/Embedding/embedding_sp_trainable_weights"):
                        continue
                    print("restore:", v.name)
                    restore_variables[v.name] = v
                saver = tf.train.Saver(restore_variables)
                saver.restore(self._sess, model_file)

            self._fetches = {
                "topk": self._language_model_test._top_k_prediction,
                "probability": self._language_model_test._probabilities,
                "final_state": self._language_model_test.final_state
            }

    def predict(self, sentence, k):
        state = self._sess.run(self._language_model_test.initial_state)
        inputs, word_letters = self._data_utility.sentence2ids(sentence)
        for i in range(len(inputs)):
            vals = self._sess.run(self._fetches, feed_dict={self._language_model_test.initial_state: state,
                                                            self._language_model_test.input_data: [[inputs[i]]],
                                                            self._language_model_test.target_data: [[0]],
                                                            self._language_model_test.output_masks: [[0.0]],
                                                            self._language_model_test.top_k: k})
            state = vals["final_state"]
        topk = vals["topk"][0]
        probability = vals["probability"][0]
        probability_topk = [probability[id] for id in topk]
        words_out = self._data_utility.ids2outwords(topk)
        return [{'word': word, 'probability': float(probability)}
                if word != '<unk>' else {'word': '<' + word_letters + '>', 'probability': float(probability)}
                for word, probability in zip(words_out, probability_topk)] if len(words_out) > 0 else []

    def predict_data(self, sentence):
        sentence = sentence.rstrip()
        state = self._sess.run(self._language_model_test.initial_state)
        inputs, words_num, letters_num = self._data_utility.data2ids_line(sentence)
        if inputs == None:
            return None
        words_out = []
        for i in range(len(inputs)):
            vals = self._sess.run(self._fetches, feed_dict={self._language_model_test.initial_state: state,
                                                            self._language_model_test.input_data: [[inputs[i]]],
                                                            self._language_model_test.target_data: [[0]],
                                                            self._language_model_test.output_masks: [[0.0]],
                                                            self._language_model_test.top_k: 3})
            state = vals["final_state"]
            top3 = vals["topk"][0]
            words = self._data_utility.ids2outwords(top3)
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
        state = self._sess.run(self._language_model_test.initial_state)
        inputs, words_num, letters_num = self._data_utility.data2ids_line(sentence)
        if inputs == None:
            return None
        words_out = []
        probability_out = []
        for i in range(len(inputs)):
            vals = self._sess.run(self._fetches, feed_dict={self._language_model_test.initial_state: state,
                                                            self._language_model_test.input_data: [[inputs[i]]],
                                                            self._language_model_test.target_data: [[0]],
                                                            self._language_model_test.output_masks: [[0.0]],
                                                            self._language_model_test.top_k: 3})
            state = vals["final_state"]
            top3 = vals["topk"][0]
            probability = vals["probability"][0]
            probability_top3 = [probability[id] for id in top3]
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

    def save_model(self, out_path):
        tf.train.write_graph(self._sess.graph_def, out_path, "graph_rnn.pb", False)

if __name__ == "__main__":
    model_path = "/home/gaoxin/workspace/python/dl-tensorflow-dev/seq2word_word_letter_sparse/resource/model/201612_201708_cleaned_20000_08_6000W_out2unk_bucket/"
    model_name = "model_test.ckpt-7427480"
    config_name = "20032_20002_400_2000_10_bucket.cfg"
    engine = InputEngineRnn(model_path, model_name, config_name)

    test_file_in = "resource/test_data/test_data_cleaned_50031"
    test_file_out = "target/201612_201708_cleaned_20000_08_6000W_out2unk_bucket_7427480_cleaned_50031"
    engine.predict_file(test_file_in, test_file_out)

    # test_file_in = "resource/test_data/test_data_50189"
    # test_file_out = "target/201612_201708_20000_08_6000W_bucket_7441850_50189_probability"
    # engine.predict_file_probability(test_file_in, test_file_out)

    # model_path = "/home/gaoqin/pubsrv/gaoqin/output/2016_2017_0_8_2UNK/model-output-400_2000_10/pretrain"
    # model_name = "model_test.ckpt-2477850"
    # config_name = "19761_19472_400_2000_10.cfg"
    # test_file_in = "/home/gaoqin/pubsrv/data_for_gaoqin/test_data_50189"
    # test_file_out = "/home/gaoqin/pubsrv/gaoqin/output/2016_2017_0_8_2UNK/model-output-400_2000_10/test.out.50189"
    # full_vocab = "/home/gaoqin/pubsrv/data_for_gaoqin/main_en_xx_unigram"
    # engine = InputEngineRnn(model_path, model_name, config_name, full_vocab)
    # engine.predict_file(test_file_in, test_file_out)

    # model_path = "/home/gaoxin/workspace/python/dl-tensorflow-dev/seq2word_word_letter_sparse/resource/model/2016_2017_10000_08_2000W_ids_combine_fix_400/"
    # model_name = "model_test.ckpt-490540"
    # config_name = "10000_400_2000_10.cfg"
    # test_file_in = "resource/test_data/test_data_10000_50189"
    # test_file_out = "target/2016_2017_10000_08_2000W_ids_combine_fix_400_490540_50189_probability"
    # engine = InputEngineRnn(model_path, model_name, config_name)
    # engine.predict_file_probability(test_file_in, test_file_out)

    # sentence = "I cant"
    # res = engine.predict(sentence, 10)
    # print(sentence)
    # print(res)

    # sentence = "you not hungry your food has 	<b> d r u g s </b>		drugs"
    # res = engine.predict_data(sentence)
    # print (sentence)
    # print (res)
    #
    # sentence = "you not hungry your food has 	<b> d r u g </b>		drugs"
    # res = engine.predict_data(sentence)
    # print(sentence)
    # print (res)
    #
    # sentence = " 	<b> d r u g </b>		drugs"
    # res = engine.predict_data(sentence)
    # print(sentence)
    # print(res)
    #
    # sentence = "you not hungry your food has drug"
    # res = engine.predict(sentence, 10)
    # print(sentence)
    # print(res)
    #
    # sentence = "happy bi"
    # res = engine.predict(sentence, 10)
    # print(sentence)
    # print(res)
    #
    # sentence = "happy mother's "
    # res = engine.predict(sentence, 10)
    # print(sentence)
    # print(res)
    #
    # out_path = "target/"
    # engine.save_model(out_path)