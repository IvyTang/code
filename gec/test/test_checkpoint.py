#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import time
from data_utility_dynamic import DataUtility
from config import Config
import tensorflow as tf
from model import Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
import numpy as np
from spacy_lemmatizer import Lemmatizer


class InputEngineRnn:

    def __init__(self, model_file, vocab_path, config_name):

        vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")
        vocab_file_in_letters = os.path.join(vocab_path, "vocab_in_letters")
        vocab_file_out = os.path.join(vocab_path, "vocab_out")
        vocab_file_lemma = os.path.join(vocab_path, "vocab_lemma")

        self._config = Config()
        self._config.get_config(vocab_path, config_name)
        self._data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words,
                                         vocab_file_in_letters=vocab_file_in_letters,
                                         vocab_file_out=vocab_file_out,
                                         vocab_file_lemma=vocab_file_lemma,
                                         vocab_freq_file="lang8_800w_vocab_freq")
        self.lemmatizer = Lemmatizer()
        self._config.batch_size = 1
        self.lemma_threshold = 0.9
        self.unk_threshold = 0.7
        self.not_unk_threshold = 0.9

        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-self._config.init_scale, self._config.init_scale)
            with tf.name_scope("Online"):
                self.model_test = Model(is_training=False, config=self._config, initializer=initializer)

            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = self._config.gpu_fraction
            self._sess = tf.Session(config=gpu_config)
            with self._sess.as_default():
                # Do not restore sparse weights from pretrain phase
                restore_variables = dict()
                for v in tf.trainable_variables():
                    print("restore:", v.name)
                    restore_variables[v.name] = v
                saver = tf.train.Saver(restore_variables)
                saver.restore(self._sess, model_file)
                self.letter_embeddings = self._sess.run(self.model_test.W)
                self.pca = PCA(n_components=2)
                # print(self.letter_embeddings.shape)
                # self.embedding_pca()

            self._fetches = {
                "topk": self.model_test.top_k_prediction,
                "probability": self.model_test.top_k_probs,
                "lemma_topk": self.model_test.lemma_top_k_prediction,
                "lemma_probability": self.model_test.lemma_top_k_probs,
                "lm_topk": self.model_test.lm_top_k_prediction,
                "lm_probability": self.model_test.lm_top_k_probs,
                "all_lm_prob": self.model_test.lm_probs,
                "lm_logits": self.model_test.lm_top_k_logits,
                "all_lm_logits": self.model_test.word_logits
            }

    def predict(self, sentence, k):
        global probabilities, top_k_predictions

        input_words, input_words_lemma, input_words_pos, inputs_ids, lemma_inputs_ids, inputs_key_ids, differ_index = \
            self._data_utility.sentence2ids(sentence)
        cnn_words_out = []
        words_out = []
        probs_out = []
        lemma_probs_out = []
        lemma_words_out = []
        lm_words_out = []
        log_prob_sentence, log_prob_sentence1 = None, None
        feed_values = {self.model_test.lm_input_data: [inputs_ids],
                       self.model_test.lemma_input_data: [lemma_inputs_ids],
                       self.model_test.input_x: inputs_key_ids,
                       self.model_test.kc_top_k: [k]}
        vals = self._sess.run(self._fetches, feed_dict=feed_values)

        top_k_predictions = vals["topk"]
        probabilities = vals["probability"]

        lemma_top_k_predictions = vals["lemma_topk"]
        lemma_probabilities = vals["lemma_probability"]

        lm_top_k_predictions = vals["lm_topk"]
        lm_probabilities = vals["lm_probability"]
        all_lm_probabilities = vals["all_lm_prob"]
        log_prob_list = []

        for (prob, input_id) in zip(all_lm_probabilities, inputs_ids[1:-1]):
            if input_id == 2:
                log_prob_list = []
                break
            log_prob = - math.log(prob[input_id])
            log_prob_list.append(log_prob)
        # print(log_prob_list)
        if len(log_prob_list) != 0:
            log_prob_sentence = np.sum(log_prob_list)/len(log_prob_list)
        # print(log_prob_sentence)

        for i in range(len(input_words)):
            probs_list = probabilities[i]
            words_list = self._data_utility.ids2outwords(top_k_predictions[i])
            lemma_probs_list = lemma_probabilities[i]
            lemma_words_list = self._data_utility.ids2outwords(lemma_top_k_predictions[i])
            lm_words_list = self._data_utility.ids2outwords(lm_top_k_predictions[i])
            lm_words_out.append(lm_words_list)

            word = words_list[0]
            prob = probs_list[0]
            lemma_word = lemma_words_list[0]

            lemma_prob = lemma_probs_list[0]

            lemma_probs_out.append(str(lemma_prob))
            probs_out.append(str(prob))
            cnn_words_out.append(word)
            if lemma_word == "<unk>" or lemma_word == "<num>" or lemma_word == "<pun>" or lemma_word == "<und>":
                lemma_words_out.append("<unk>")

            elif lemma_word != input_words[i] and lemma_word:
                lemma_word_orig = self.lemmatizer.lemmatize(lemma_word)[1][0]

                if lemma_prob >= self.lemma_threshold and lemma_word_orig == input_words_lemma[i]:
                    lemma_words_out.append(lemma_word)
                else:
                    lemma_words_out.append("<unk>")
            else:
                lemma_words_out.append("<unk>")

            if word == "<unk>" or word == "<num>" or word == "<pun>" or word == "<und>":
                words_out.append(input_words[i])

            elif input_words[i] not in self._data_utility.token2id_in_words \
                    and input_words[i].lower() not in self._data_utility.token2id_in_words \
                    and input_words[i].lower().capitalize() not in self._data_utility.token2id_in_words:
                if prob >= self.unk_threshold and word != input_words[i]:
                    words_out.append(word)
                else:
                    words_out.append(input_words[i])

            elif input_words[i] in self._data_utility.token2id_in_words \
                    or input_words[i].lower() in self._data_utility.token2id_in_words\
                    or input_words[i].lower().capitalize() in self._data_utility.token2id_in_words:

                if prob >= self.not_unk_threshold and word != input_words[i]:
                    words_out.append(word)
                else:
                    words_out.append(input_words[i])

            else:
                words_out.append(input_words[i])
        if len(differ_index) > 0:
            for index in differ_index:
                if lemma_words_out[index] != "<unk>":
                    words_out[index] = lemma_words_out[index]
                    probs_out[index] = lemma_probs_out[index]
 
        res_inputs_ids = self._data_utility.words2ids(words_out)
        feed_values = {self.model_test.lm_input_data: [res_inputs_ids]}
        fetches = {"lm_topk": self.model_test.lm_top_k_prediction,
                "lm_probability": self.model_test.lm_top_k_probs,
                "all_lm_prob": self.model_test.lm_probs,
                "lm_logits": self.model_test.lm_top_k_logits,
                "all_lm_logits": self.model_test.word_logits
            }
        vals = self._sess.run(fetches, feed_dict=feed_values)

        all_lm_probabilities = vals["all_lm_prob"]
        log_prob_list = []

        for (prob, res_input_id, input_id) in zip(all_lm_probabilities, res_inputs_ids[1:-1], inputs_ids[1:-1]):
            if input_id == 2:
                log_prob_list = []
                break
            log_prob = - math.log(prob[res_input_id])
            log_prob_list.append(log_prob)
        if len(log_prob_list) != 0:
            log_prob_sentence1 = np.sum(log_prob_list) / len(log_prob_list)

        return input_words, input_words_lemma, \
               words_out, cnn_words_out, \
               probs_out, lemma_words_out, \
               lm_words_out, lm_probabilities, \
               log_prob_list, log_prob_sentence

    def embedding_pca(self):
        self.pca.fit(self.letter_embeddings)
        print(self.pca.explained_variance_ratio_)
        print(self.pca.explained_variance_)
        X_new = self.pca.transform(self.letter_embeddings)
        for i in range(self._data_utility.in_letters_count):
            plt.scatter(X_new[i, 0], X_new[i, 1], label=self._data_utility.id2token_in_letters[i])
            plt.annotate(
                '%s' % self._data_utility.id2token_in_letters[i],
                xy=(X_new[i, 0], X_new[i, 1]),
                xytext=(0, -10),
                textcoords='offset points',
                ha='center',
                va='top')
        plt.show()

    def predict_data(self, sentence, k, min_prob, lemma_min_prob):
        sentence = sentence.rstrip()
        data_and_ids = self._data_utility.data2ids_line(sentence)
        if data_and_ids is None:
            return None

        sentence_original, sentence_lemmatized, letters_line, words_ids, lemma_words_ids, letters_ids, \
                                words_num, lemmatized_idx= data_and_ids

        words_out = []
        probs_out = []
        lemma_probs_out = []
        lemma_words_out = []

        feed_values = {self.model_test.lm_input_data: [words_ids],
                       self.model_test.lemma_input_data: [lemma_words_ids],
                       self.model_test.input_x: letters_ids,
                       self.model_test.kc_top_k: [k]}
        vals = self._sess.run(self._fetches, feed_dict=feed_values)
        top_k_predictions = vals["topk"]
        probabilities = vals["probability"]
        lemma_top_k_predictions = vals["lemma_topk"]
        lemma_probabilities = vals["lemma_probability"]

        for i in range(len(sentence_original)):
            probs_list = probabilities[i]
            words_list = self._data_utility.ids2outwords(top_k_predictions[i])
            lemma_probs_list = lemma_probabilities[i]
            lemma_words_list = self._data_utility.ids2outwords(lemma_top_k_predictions[i])
            word = words_list[0]
            prob = probs_list[0]
            lemma_word = lemma_words_list[0]
            lemma_prob = lemma_probs_list[0]

            probs_out.append(str(prob))
            lemma_probs_out.append(str(lemma_prob))

            if lemma_word == "<unk>" or lemma_word == "<num>" or lemma_word == "<pun>" or lemma_word == "<und>":
                lemma_words_out.append("<unk>")

            elif lemma_word != letters_line[i]:
                if lemma_prob >= lemma_min_prob:
                    lemma_words_out.append(lemma_word)
                else:
                    lemma_words_out.append("<unk>")
            else:
                lemma_words_out.append("<unk>")

            if word == "<unk>" or word == "<num>" or word == "<pun>" or word == "<und>":
                words_out.append(letters_line[i])

            elif word != letters_line[i]:
                if prob >= min_prob:
                    words_out.append(word)
                else:
                    words_out.append(letters_line[i])
            else:
                words_out.append(word)

        if len(lemmatized_idx) > 0:
            for index in lemmatized_idx:
                if lemma_words_out[index] != "<unk>":
                    words_out[index] = lemma_words_out[index]
                    probs_out[index] = lemma_probs_out[index]

        return sentence_original, sentence_lemmatized, letters_line, words_out, probs_out

    def predict_data_cnn(self, sentence, k, min_prob):
        sentence = sentence.rstrip()
        raw_line, raw_inputs, raw_inputs_key, inputs, inputs_key, words_num, letters_num = \
            self._data_utility.data2ids_line_cnn(sentence)
        probability_topk = list()
        words_out = list()
        feed_values = {self.model_test.lm_input_data: [inputs],
                       self.model_test.lemma_input_data: [inputs[1:-1]],
                       self.model_test.input_x: inputs_key,
                       self.model_test.kc_top_k: [k]}
        vals = self._sess.run(self._fetches, feed_dict=feed_values)

        top_k_predictions = vals["topk"]
        probabilities = vals["probability"]

        for i in range(words_num):
            probs_list = [prob for prob in probabilities[i]]
            words_list = self._data_utility.ids2outwords(top_k_predictions[i])
            word = words_list[0]
            prob = probs_list[0]
            replaced_word = raw_inputs_key[i]

            if word == "<unk>" or word == "<num>" or word == "<pun>" or word == "<und>":
                words_out.append(replaced_word)
            elif word != replaced_word:
                if prob >= min_prob:
                    words_out.append(word)
                else:
                    words_out.append(replaced_word)

            else:
                words_out.append(word)

            probability_topk.append(str(prob))

        return raw_line, raw_inputs_key, words_out, probability_topk

    def result_print(self, out_string, out_prob):
        string = ""
        for (word, prob) in zip(out_string, out_prob):
            prob = str(prob) if word != "" else "0.0"
            string = string + word + ":" + prob + "|"
        string = string[:-1]
        return string

    def predict_file(self, test_file_in, test_file_out, k, min_prob):
        testfilein = open(test_file_in, "r")
        testfileout = open(test_file_out, 'w')
        t1 = time.time()

        for sentence in testfilein:
            sentence = sentence.rstrip()
            result = self.predict_data_cnn(sentence, k, min_prob)

            if result is not None:
                output_words, input_words, out_words_list, out_prob_list = result

                testfileout.write(" ".join(output_words) + "|#|" + " ".join(input_words) + "|#|" +
                                  "|".join([":".join([word, prob]) for (word, prob) in
                                            zip(out_words_list, out_prob_list)]) + "\n")

        t2 = time.time()
        print(t2 - t1)
        testfilein.close()
        testfileout.close()

    def predict_test(self, test_file_in, k, test_file_out):
        testfilein = open(test_file_in, "r")

        with open(test_file_out, "w") as f:
            for sentence in testfilein:
                sentence = sentence.rstrip()
                origin_words, origin_words_lemma, \
                words_out, cnn_words_out, \
                probs_out, _, _, _,\
                log_list, log_sentence = \
                    self.predict(sentence, k)

                words_out = " ".join(words_out)

                print(words_out)
                f.write(words_out + "\n")
        testfilein.close()
        f.close()


if __name__ == "__main__":
    args = sys.argv

    model_file = args[1]
    vocab_path = args[2]
    config_name = args[3]
    test_file_in = args[4]
    test_file_out = args[5]

    engine = InputEngineRnn(model_file, vocab_path, config_name)
    engine.predict_test(test_file_in, 3, test_file_out)