# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import os
import random
import sys
from compressor import Compressor
import multiprocessing
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from data_feeder import DataFeederContext
from data_utility import DataUtility

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model_config", "static-sanity-check.cfg", "Model hyperparameters configuration. See smallconfig.cfg for an example.")
flags.DEFINE_string("mode", "pretrain", "How to use this model. Possible options are: pretrain, learn_basis, finetune, inference")
flags.DEFINE_string("data_path", "resource/train_data/", "Where the training/test data is stored.")
flags.DEFINE_string("vocab_path", "resource/vocab/", "Where the training/test data is stored.")
flags.DEFINE_string("file_name_stem", "ptb", "corpus file name (without suffix).")
flags.DEFINE_string("num_test_cases", 0, "# of test cases reading from stdin. Works for inference phase only. Default: zero.")
flags.DEFINE_string("save_path", "target/model/", "Model output directory.")
flags.DEFINE_string("model_name", "model_test", "Pick a name for your model.")
flags.DEFINE_string("graph_save_path", "target/graph/", "Exported graph directory.")
flags.DEFINE_integer("laptop_discount", 1, "Run less iterations on laptop.")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("cpu_count", 2, "# of cpus to calculate sparse representation in parallel.")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def index_data_type():
    return tf.int32


def np_index_data_type():
    return np.int32


class PTBModel(object):
    """Static PTB model. Modified from old saniti-checked version of dynamic model.
    """

    def __init__(self, is_training, config):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.size = config.hidden_size
        self.vocab_size_in = config.vocab_size_in
        self.vocab_size_out = config.vocab_size_out
        self.basis_size = config.basis_size
        self.sparsity = config.sparsity
        self.batch_input_indices_to_feed = []
        self.softmax_indices_to_feed = []

        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None], name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None], name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None], name="batched_output_word_masks")
        # self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="batched_input_sequence_length")
        # self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size], value=self.num_steps),
        #                                                    shape=[self.batch_size], name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")

        # total number of words in this mini-batch (assume all sentences are of same lengths)
        num_words = tf.size(self.input_data)

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        # input state is a regular Tensor,
        # which needs to be converted to LSTMStateTuple to be fed to tf.nn.static_rnn(),
        # which generates a regular Tensor thus can be fed back for the next batch.
        lstm_state_as_tensor_shape = [config.num_layers, 2, config.batch_size, config.hidden_size]
        self._initial_state = tf.placeholder_with_default(tf.zeros(lstm_state_as_tensor_shape, dtype=data_type()),
                                                          lstm_state_as_tensor_shape, name="state")
        unstack_state = tf.unstack(self._initial_state, axis=0)
        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1])
             for idx in range(config.num_layers)]
        )

        with tf.variable_scope("Embedding"):
            self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.size], dtype=data_type())
            self._embedding_basis = tf.get_variable("embedding_basis", [self.basis_size, self.size], dtype=data_type())

            # Use normal embedding in pretrain phase, use sparse embedding in finetune or inference phase

            if FLAGS.mode == "pretrain" or FLAGS.mode == "learn_basis":
                inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)
            else:
                # Finetune or inference phase
                # restore sparse embedding parameters from checkpoints

                # `basis_to_gather` contains 1-D indices from which id values and weights values should be gathered.
                # For example, for a mini-batch [[3, 1, 2], [5, 2, 8]]
                # basis_to_gather is:
                #       [3 * sparsity + 0, 3 * sparsity + 1, ..., 3 * sparsity + sparsity - 1,
                #        1 * sparsity + 0, 1 * sparsity + 1, ..., 1 * sparsity + sparsity - 1,
                #           ..., 8 * sparsity + 0, 8 * sparsity + 1, ..., 8 * sparsity + sparsity - 1]

                # idx2 = [0, 1, 2, ..., sparsity-1, 0, 1, 2, ..., sparsity-1, ..., 0, 1, 2, ..., sparsity-1]
                idx2 = tf.tile(tf.range(self.sparsity), [num_words])

                input_data_as_a_column = tf.reshape(self.input_data, [-1, 1])

                # idx3 = [3, 3, ..., 3, 1, 1, ..., 1, ..., 8, 8, ..., 8]
                idx3 = tf.reshape(tf.tile(input_data_as_a_column, [1, self.sparsity]), [-1])

                basis_to_gather = idx3 * self.sparsity + idx2

                finetune_save_path = os.path.join(FLAGS.save_path, "finetune-" + FLAGS.model_config)
                if not os.path.isdir(finetune_save_path):
                    os.mkdir(finetune_save_path)
                sparse_parameters_path = finetune_save_path

                embedding_sp_ids_val = np.load(os.path.join(sparse_parameters_path, "embedding_sp_ids_val.npy"))
                embedding_sp_weights_val = np.load(os.path.join(sparse_parameters_path, "embedding_sp_weights_val.npy"))

                embedding_init = tf.constant_initializer(embedding_sp_weights_val)
                embedding_sp_trainable_weights = tf.get_variable(shape=[embedding_sp_weights_val.shape[0]],
                                                                 dtype=data_type(),
                                                                 name="embedding_sp_trainable_weights",
                                                                 initializer=embedding_init)

                # basis_ids and basis_weights are all 1-D tensors
                basis_ids = tf.gather(embedding_sp_ids_val, basis_to_gather)
                basis_weights = tf.gather(embedding_sp_trainable_weights, basis_to_gather)

                # Find all basis embeddings required by this mini-batch
                # collected_embeddings is a 2-D tensor of shape (batch_size * time_step * sparsity) * hidden_size
                collected_embeddings = tf.nn.embedding_lookup(self._embedding_basis, basis_ids)
                # Multiply each basis vector by weight, then group every sparsity rows and sum them up
                inputs = tf.reduce_sum(tf.reshape(collected_embeddings * tf.reshape(basis_weights, [-1, 1]),
                                                  [-1, self.sparsity, self.size]), axis=1)

                # Now, inputs.shape = (batch_size * time_step) * hidden_size
                # Reshape it back to 3D batch_size * time_step * hidden_size
                inputs = tf.reshape(inputs, [self.batch_size, -1, self.size])

            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

        with tf.variable_scope("RNN"):
            inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
            # outputs, state_out = tf.contrib.rnn.static_rnn(cell, inputs, sequence_length=self.sequence_length, initial_state=tuple_state)
            outputs, state_out = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=tuple_state)
            # outputs, state_out = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.seqlen, initial_state=tuple_state)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.size])
        with tf.variable_scope("Softmax"):
            # Now output.shape = (batch_size * step_size) * hidden_size
            self._softmax_w = tf.get_variable("softmax_w", [self.size, self.vocab_size_out], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())
            self._softmax_w_basis = tf.get_variable("softmax_w_basis", [self.size, self.basis_size], dtype=data_type(),
                                                    initializer=tf.zeros_initializer(dtype=data_type()))

            # Use normal softmax in pretrain phase, use sparse softmax in finetune or inference phase
            if FLAGS.mode == "pretrain" or FLAGS.mode == "learn_basis":
                logits = tf.matmul(output, self._softmax_w) + softmax_b
            else:
                # Finetune or inference phase
                # restore_sparse_softmax_parameters from checkpoints

                finetune_save_path = os.path.join(FLAGS.save_path, "finetune-" + FLAGS.model_config)
                if not os.path.isdir(finetune_save_path):
                    os.mkdir(finetune_save_path)

                # sparse_parameters_path = os.path.join(finetune_save_path, FLAGS.model_name)
                sparse_parameters_path = finetune_save_path
                softmax_sp_ids_val = np.load(os.path.join(sparse_parameters_path, "softmax_sp_ids_val.npy"))
                softmax_ids = tf.constant(softmax_sp_ids_val, dtype=index_data_type())

                softmax_sp_weights_val = np.load(os.path.join(sparse_parameters_path, "softmax_sp_weights_val.npy"))

                softmax_init = tf.constant_initializer(softmax_sp_weights_val)
                softmax_sp_trainable_weights = tf.get_variable(shape=[softmax_sp_weights_val.shape[0]],
                                                               dtype=data_type(),
                                                               name="softmax_sp_trainable_weights",
                                                               initializer=softmax_init)
                # print(softmax_sp_trainable_weights.name)

                # Denote batch_size * time_step by batch_size'
                # logits_basis.shape = (batch_size' * hidden_size) x (hidden_size * basis_size) -> batch_size' * basis_size
                logits_basis = tf.matmul(output, self._softmax_w_basis)
                logits_basis_T = tf.transpose(logits_basis)  # basis_size * batch_size'

                # Find all basis embeddings required by this mini-batch
                # collected_embeddings is a 2-D tensor of shape (vocab_size_out * sparsity) * batch_size'
                collected_embeddings = tf.nn.embedding_lookup(logits_basis_T, softmax_ids)

                # Multiply each basis vector by weight, then group every sparsity rows and sum them up
                logits_full_T = tf.reduce_sum(tf.reshape(collected_embeddings * tf.reshape(softmax_sp_trainable_weights, [-1, 1]),
                                                         [self.vocab_size_out, self.sparsity, -1]), axis=1)
                logits = tf.transpose(logits_full_T) + softmax_b  # batch_size' * vocab_size_out

        # probabilities.shape = (batch_size * time_step) * vocab_size_out
        probabilities = tf.nn.softmax(logits, name="probabilities")
        print(probabilities.name)
        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")
        print(top_k_prediction.name)  # "Online/Model/top_k_prediction:1"

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.output_masks, [-1])],
                                                                  average_across_timesteps=False)

        # [tf.ones([self.batch_size * self.num_steps], dtype=data_type())])
        # self._cost = cost = tf.reduce_sum(loss) / self.batch_size
        self._cost = cost = tf.reduce_sum(loss)
        self._final_state = tf.identity(state_out, "state_out")  # Use tf.identity to take a name for final state
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        # Ops setting sparse softmax params
        self._new_softmax_basis = tf.placeholder(data_type(), shape=[self.size, self.basis_size])
        self._softmax_basis_update = tf.assign(self._softmax_w_basis, self._new_softmax_basis)

        # Ops setting sparse embedding params
        self._new_embedding_basis = tf.placeholder(data_type(), shape=[self.basis_size, self.size])
        self._embedding_basis_update = tf.assign(self._embedding_basis, self._new_embedding_basis)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_sparse_softmax_params(self, session, basis):
        session.run(self._softmax_basis_update, feed_dict={self._new_softmax_basis: basis})

    def assign_sparse_embedding_params(self, session, basis):
        session.run(self._embedding_basis_update, feed_dict={self._new_embedding_basis: basis})

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def softmax_w(self):
        return self._softmax_w

    @property
    def cost(self):
        return self._cost

    @property
    def embedding(self):
        return self._embedding

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def logits(self):
        return self._logits

    @property
    def probalities(self):
        return self._probabilities

    @property
    def top_k_prediction(self):
        return self._top_k_prediction

    @property
    def train_op(self):
        return self._train_op


def export_graph(session):
    graph_def = convert_variables_to_constants(session, session.graph_def,
                                               ["Online/Model/probabilities", "Online/Model/state_out", "Online/Model/top_k_prediction"])
    model_export_name = os.path.join(FLAGS.graph_save_path, 'sparse_graph-' + FLAGS.mode + '-' + FLAGS.model_config + '.pb')
    f = open(model_export_name, "wb")
    f.write(graph_def.SerializeToString())
    f.close()
    print("Graph is saved to: ", model_export_name)


def run_evaluate_epoch(session, model, logfile, word_dict=None, data_feeder=None, eval_op=None, output_limit=10000):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0

    fetches = {
        "cost": model.cost,
        "top_k_prediction": model.top_k_prediction
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    batch_size = model.batch_size
    num_steps = model.num_steps
    epoch_size = data_feeder.num_samples // batch_size

    # To prevent prediction overflow
    prediction_made = 0.0
    top1_correct_total, top3_correct_total, top5_correct_total = 0.0, 0.0, 0.0

    for step in range(epoch_size):
        inputs, outputs, masks, lengths = data_feeder.next_batch_fixmask(batch_size)
        feed_dict = {model.input_data: inputs,
                     model.target_data: outputs,
                     model.output_masks: masks,
                     # model.sequence_length: lengths,
                     model.top_k: 5}

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        costs += cost
        iters += np.sum(masks)

        # top_k_prediction.shape = (batch_size * num_steps) * 5 (# of top_k items to extract)
        top_k_prediction = vals["top_k_prediction"]  # we take care of indices only (do not care about logits)
        y_as_a_column = outputs.reshape([-1])

        num_prediction_made_in_batch = len(y_as_a_column)
        top1_correct = np.sum((top_k_prediction[:, 0] == y_as_a_column).astype(float))
        top3_correct = top1_correct + np.sum((top_k_prediction[:, 1] == y_as_a_column).astype(float)) \
                       + np.sum((top_k_prediction[:, 2] == y_as_a_column).astype(float))
        top5_correct = top3_correct + np.sum((top_k_prediction[:, 3] == y_as_a_column).astype(float)) \
                       + np.sum((top_k_prediction[:, 4] == y_as_a_column).astype(float))

        top1_correct_total += top1_correct
        top3_correct_total += top3_correct
        top5_correct_total += top5_correct

        if prediction_made < output_limit:
            if word_dict is not None:
                for i in range(num_prediction_made_in_batch):
                    print("candidates:", ", ".join([word_dict[word] for word in top_k_prediction[i]]),
                          "desired:", word_dict[int(y_as_a_column[i])], file=logfile)
            else:
                for i in range(num_prediction_made_in_batch):
                    print("candidates: ", top_k_prediction, "; desired: ", outputs, file=logfile)
        prediction_made += batch_size * num_steps

    # Prediction accuracy information:
    print("Top1 accuracy = {0}, top3 accuracy = {1}, top5 accuracy = {2} ".format(top1_correct_total / prediction_made,
                                                                                  top3_correct_total / prediction_made,
                                                                                  top5_correct_total / prediction_made),
          file=logfile)
    end_time = time.time()
    print("Test time = {0}".format(end_time - start_time))
    return np.exp(costs / iters)


def run_epoch(session, model, eval_op=None, data_feeder=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    num_word = 0

    fetches = {
        "cost": model.cost,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    batch_size = model.batch_size
    num_steps = model.num_steps
    epoch_size = data_feeder.num_samples // batch_size

    # for debug purpose
    for step in range(epoch_size):
        if step >= epoch_size // FLAGS.laptop_discount:  # Early stopping for debug purpose
            break
        inputs, outputs, masks, lengths = data_feeder.next_batch_fixmask(batch_size)
        feed_dict = {model.input_data: inputs,
                     model.target_data: outputs,
                     model.output_masks: masks}
                     # model.sequence_length: lengths}

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]

        costs += cost
        iters += np.sum(masks)
        num_word += np.sum(lengths)

        if verbose and step % (epoch_size // 100) == 0:
            if costs / iters > 100.0:
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] PPL TOO LARGE! %.3f ENTROPY: (%.3f) speed: %.0f wps" %
                      (step * 1.0 / epoch_size, costs / iters, num_word / (time.time() - start_time)))
            else:
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] %.3f perplexity: %.3f speed: %.0f wps" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters), num_word / (time.time() - start_time)))
            sys.stdout.flush()

    return np.exp(costs / iters)


def fit_wrapper(compressor_x):
    compressor, x = compressor_x
    return compressor.fit(x)


def learn_sparse_softmax(session, model, verbose=False):
    # Each vector has model.sparsity non-zero entries!

    softmax_param = session.run(model.softmax_w)
    softmax_w_basis = softmax_param[:, :model.basis_size]  # Choose first model.basis_size columns from softmax_w
    model.assign_sparse_softmax_params(session, softmax_w_basis)

    softmax_sp_ids_val = np.zeros(model.vocab_size_out * model.sparsity, dtype=np_index_data_type())
    softmax_sp_weights_val = np.zeros(model.vocab_size_out * model.sparsity)

    # print("max non zero entry: ", model.sparsity)
    # columns are bases. No need to transpose.
    compressor = Compressor(bases=softmax_w_basis, max_non_zero_entry=model.sparsity)

    # For reference:
    #     sp_indices: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 1], [3, 1]],
    #     sp_shape: [4, 3],  # batch size: 4, max index: 2 (so index count == 3)
    #     sp_ids_val: [5, 8, 1, 0, 6, 2, 7],
    #     sp_weights_val: np.array([0.1, 0.2, 0.5, -1.0, 0.5, 1.0, 0.8])})

    # For basis vectors, they have one-hot expressions
    for i in range(model.basis_size):
        random_basis = {i: True}
        j = 1  # j: number of elements in random_basis
        while j < model.sparsity:
            r = random.randint(0, model.basis_size - 1)
            if (r not in random_basis):
                random_basis[r] = True
                j += 1
        del random_basis[i]
        # 1 true basis, model.sparsity - 1 paddings
        indices = np.r_[np.array([i], dtype=np_index_data_type()),
                        np.array(random_basis.keys(), dtype=np_index_data_type())]
        softmax_sp_ids_val[i * model.sparsity: (i + 1) * model.sparsity] = indices
        # other weights are zero by the definition of `softmax_sp_weights_val`
        softmax_sp_weights_val[i * model.sparsity] = 1.0

    t1 = time.time()
    pool = multiprocessing.Pool(processes=FLAGS.cpu_count)
    softmax_param = softmax_param.T
    parallel_params = [(compressor, softmax_param[i, :]) for i in range(model.basis_size, model.vocab_size_out)]
    results = pool.map(fit_wrapper, parallel_params)
    for i_, result in enumerate(results):
        indices, values = result
        i = i_ + model.basis_size
        softmax_sp_ids_val[i * model.sparsity: (i + 1) * model.sparsity] = indices
        softmax_sp_weights_val[i * model.sparsity: (i + 1) * model.sparsity] = values

    t2 = time.time()

    print("Parallel time: ", t2 - t1)
    # print(softmax_sp_ids_val[:20])
    # print(softmax_sp_ids_val[-20:])
    # print(softmax_sp_weights_val[:20])
    # print(softmax_sp_weights_val[-20:])

    finetune_save_path = os.path.join(FLAGS.save_path, "finetune-" + FLAGS.model_config)
    if not os.path.isdir(finetune_save_path):
        os.mkdir(finetune_save_path)

    sparse_parameters_path = finetune_save_path

    np.save(os.path.join(sparse_parameters_path, "softmax_sp_ids_val.npy"), softmax_sp_ids_val)
    np.save(os.path.join(sparse_parameters_path, "softmax_sp_weights_val.npy"), softmax_sp_weights_val)
    return softmax_w_basis, softmax_sp_ids_val, softmax_sp_weights_val


def learn_sparse_embedding(session, model, verbose=False, output_frequency=500):
    embedding_param = session.run(model.embedding)
    embedding_basis = embedding_param[:model.basis_size, :]  # Choose first model.basis_size rows as basis

    embedding_sp_ids_val = np.zeros(model.vocab_size_in * model.sparsity, dtype=np_index_data_type())
    embedding_sp_weights_val = np.zeros(model.vocab_size_in * model.sparsity)

    # For basis vectors, they have one-hot expressions
    for i in range(model.basis_size):
        random_basis = {i: True}
        j = 1  # j: number of elements in random_basis
        while j < model.sparsity:
            r = random.randint(0, model.basis_size - 1)
            if (r not in random_basis):
                random_basis[r] = True
                j += 1
        del random_basis[i]
        # 1 true basis, model.sparsity - 1 paddings
        indices = np.r_[np.array([i], dtype=np_index_data_type()),
                        np.array(random_basis.keys(), dtype=np_index_data_type())]
        embedding_sp_ids_val[i * model.sparsity: (i + 1) * model.sparsity] = indices
        # other weights are zero by the definition of `softmax_sp_weights_val`
        embedding_sp_weights_val[i * model.sparsity] = 1.0

    t1 = time.time()

    # print("max non zero entry: ", model.sparsity)
    # columns are bases. Need to transpose.
    compressor = Compressor(bases=embedding_basis.T, max_non_zero_entry=model.sparsity)
    pool = multiprocessing.Pool(processes=FLAGS.cpu_count)
    parallel_params = [(compressor, embedding_param[i, :]) for i in range(model.basis_size, model.vocab_size_in)]
    results = pool.map(fit_wrapper, parallel_params)
    for i_, result in enumerate(results):
        indices, values = result
        i = i_ + model.basis_size
        embedding_sp_ids_val[i * model.sparsity: (i + 1) * model.sparsity] = indices
        embedding_sp_weights_val[i * model.sparsity: (i + 1) * model.sparsity] = values

    t2 = time.time()
    print("Parallel time: ", t2 - t1)

    finetune_save_path = os.path.join(FLAGS.save_path, "finetune-" + FLAGS.model_config)
    if not os.path.isdir(finetune_save_path):
        os.mkdir(finetune_save_path)
    sparse_parameters_path = finetune_save_path

    np.save(os.path.join(sparse_parameters_path, "embedding_sp_ids_val.npy"), embedding_sp_ids_val)
    np.save(os.path.join(sparse_parameters_path, "embedding_sp_weights_val.npy"), embedding_sp_weights_val)
    model.assign_sparse_embedding_params(session, embedding_basis)
    return embedding_basis, embedding_sp_ids_val, embedding_sp_weights_val


class Config():
    def __init__(self):
        # default configuration (small configuration)
        self.init_scale = 0.05
        self.learning_rate = 1.0
        self.max_grad_norm = 5
        self.num_layers = 2
        self.num_steps = 35
        self.hidden_size = 200
        self.max_epoch = 4
        self.keep_prob = 1.0
        self.lr_decay = 0.8
        self.batch_size = 20
        self.vocab_size_in = 5000
        self.vocab_size_out = 5000
        self.max_max_epoch = 16
        self.basis_size = 600
        self.sparsity = 10
        self.finetune_learning_rate = 0.2
        self.finetune_epoch = 8
        self.gpu_fraction = 0.32

    def get_config(self, config_filename=None):
        if config_filename is not None:
            with open(config_filename) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    param, value = line.split()
                    if param == "init_scale":
                        self.init_scale = float(value)
                    elif param == "learning_rate":
                        self.learning_rate = float(value)
                    elif param == "max_grad_norm":
                        self.max_grad_norm = float(value)
                    elif param == "num_layers":
                        self.num_layers = int(value)
                    elif param == "num_steps":
                        self.num_steps = int(value)
                    elif param == "hidden_size":
                        self.hidden_size = int(value)
                    elif param == "max_epoch":
                        self.max_epoch = int(value)
                    elif param == "max_max_epoch":
                        self.max_max_epoch = int(value)
                    elif param == "keep_prob":
                        self.keep_prob = float(value)
                    elif param == "lr_decay":
                        self.lr_decay = float(value)
                    elif param == "batch_size":
                        self.batch_size = int(value)
                    elif param == "vocab_size_in":
                        self.vocab_size_in = int(value)
                    elif param == "vocab_size_out":
                        self.vocab_size_out = int(value)
                    elif param == "basis_size":
                        self.basis_size = int(value)
                    elif param == "sparsity":
                        self.sparsity = int(value)
                    elif param == "finetune_learning_rate":
                        self.finetune_learning_rate = float(value)
                    elif param == "finetune_epoch":
                        self.finetune_epoch = int(value)
                    elif param == "gpu_fraction":
                        self.gpu_fraction = float(value)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    logfile = open(FLAGS.mode + '-' + FLAGS.model_config + '.log', 'w')
    # logfile = sys.stdout

    if not os.path.isdir(FLAGS.save_path):
        os.mkdir(FLAGS.save_path)
    if not os.path.isdir(FLAGS.graph_save_path):
        os.mkdir(FLAGS.graph_save_path)

    config = Config()
    config.get_config(FLAGS.model_config)

    test_config = Config()
    test_config.get_config(FLAGS.model_config)
    test_config.batch_size = 1
    test_config.num_steps = 1

    vocab_file_in_words = os.path.join(FLAGS.vocab_path, "vocab_in_words")
    vocab_file_in_letters = os.path.join(FLAGS.vocab_path, "vocab_in_letters")
    vocab_file_out = os.path.join(FLAGS.vocab_path, "vocab_out")

    train_file_in_words = os.path.join(FLAGS.data_path, "train_in_ids_words")
    train_file_in_letters = os.path.join(FLAGS.data_path, "train_in_ids_letters")
    train_file_out = os.path.join(FLAGS.data_path, "train_out_ids")
    dev_file_in_words = os.path.join(FLAGS.data_path, "dev_in_ids_words")
    dev_file_in_letters = os.path.join(FLAGS.data_path, "dev_in_ids_letters")
    dev_file_out = os.path.join(FLAGS.data_path, "dev_out_ids")

    data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words, vocab_file_in_letters=vocab_file_in_letters,
                               vocab_file_out=vocab_file_out, max_sentence_length=config.num_steps)

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_fraction
        with tf.Session(config=gpu_config) as session:
            with tf.name_scope("Train"):
                train_feeder = DataFeederContext(vocab_file_in_words=vocab_file_in_words,
                                                 vocab_file_in_letters=vocab_file_in_letters,
                                                 vocab_file_out=vocab_file_out,
                                                 corpus_file_in_words=train_file_in_words,
                                                 corpus_file_in_letters=train_file_in_letters,
                                                 corpus_file_out=train_file_out,
                                                 max_sentence_length=config.num_steps)

                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    mtrain = PTBModel(is_training=True, config=config)
                tf.summary.scalar("Training Loss", mtrain.cost)
                tf.summary.scalar("Learning Rate", mtrain.lr)

            with tf.name_scope("Valid"):
                valid_feeder = DataFeederContext(vocab_file_in_words=vocab_file_in_words,
                                                 vocab_file_in_letters=vocab_file_in_letters,
                                                 vocab_file_out=vocab_file_out,
                                                 corpus_file_in_words=dev_file_in_words,
                                                 corpus_file_in_letters=dev_file_in_letters,
                                                 corpus_file_out=dev_file_out,
                                                 max_sentence_length=config.num_steps)
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mvalid = PTBModel(is_training=False, config=config)
                tf.summary.scalar("Validation Loss", mvalid.cost)

            # Evaluate on test data
            with tf.name_scope("Test"):
                test_feeder = DataFeederContext(vocab_file_in_words=vocab_file_in_words,
                                                vocab_file_in_letters=vocab_file_in_letters,
                                                vocab_file_out=vocab_file_out,
                                                corpus_file_in_words=dev_file_in_words,
                                                corpus_file_in_letters=dev_file_in_letters,
                                                corpus_file_out=dev_file_out,
                                                max_sentence_length=config.num_steps)
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mtest = PTBModel(is_training=False, config=config)

            # Model to be saved and exported
            # Note: it's beneficial to distinguish between test model and save model,
            # because when evaluating on test set, a large batch size is more GPU-friendly and faster.
            # But when running on cellphone, it can accept a batch size of 1 only, this is why monline exists.
            with tf.name_scope("Online"):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    monline = PTBModel(is_training=False, config=test_config)

            # Do not restore sparse weights from pretrain phase
            restore_variables = dict()
            for v in tf.trainable_variables():
                if v.name.startswith("Model/Softmax/softmax_sp_trainable_weights") \
                        or v.name.startswith("Model/Embedding/embedding_sp_trainable_weights"):
                    continue
                print("store:", v.name)
                restore_variables[v.name] = v

            sv = tf.train.Saver(restore_variables)
            if not FLAGS.model_name.endswith(".ckpt"):
                FLAGS.model_name += ".ckpt"

            session.run(tf.global_variables_initializer())
            if FLAGS.mode == "pretrain":
                # restore previously trained model
                check_point_dir = os.path.join(FLAGS.save_path, "pretrain")
                ckpt = tf.train.get_checkpoint_state(check_point_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    sv.restore(session, ckpt.model_checkpoint_path)
                else:
                    print("Created model with fresh parameters.")
                for i in range(config.max_max_epoch // FLAGS.laptop_discount):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
                    mtrain.assign_lr(session, config.learning_rate * lr_decay)

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)), file=logfile)
                    train_perplexity = run_epoch(session, mtrain, eval_op=mtrain.train_op, data_feeder=train_feeder, verbose=True)

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity), file=logfile)
                    logfile.flush()

                    valid_perplexity = run_epoch(session, mvalid, data_feeder=valid_feeder)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity), file=logfile)
                    logfile.flush()

                    print("save path:", FLAGS.save_path)

                    # Save model if FLAGS.mode == "pretrain" or "finetune"
                    if FLAGS.save_path:
                        print("Saving model to %s." % FLAGS.save_path, file=logfile)
                        step = mtrain.get_global_step(session)
                        pretrain_save_path = os.path.join(FLAGS.save_path, "pretrain")
                        if not os.path.isdir(pretrain_save_path):
                            os.mkdir(pretrain_save_path)
                        model_save_path = os.path.join(pretrain_save_path, FLAGS.model_name)
                        sv.save(session, model_save_path, global_step=step)

                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin exporting graph!")
                export_graph(session)  # Export dense graph
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish exporting graph!")

                # Evaluate on test data for {"pretrain", "finetune",} phase
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin exporting graph!")
                export_graph(session)  # Export dense graph

                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin test epoch!")
                sys.stdout.flush()
                print("=" * 30 + FLAGS.mode + "=" * 30, file=logfile)
                test_perplexity = run_evaluate_epoch(session, mtest, logfile, word_dict=data_utility.id2token_out, data_feeder=test_feeder)
                print("Test Perplexity: %.3f" % test_perplexity, file=logfile)
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish test epoch!")
                print("Test Perplexity: %.3f" % test_perplexity)  # print to stdout
                logfile.close()

            elif FLAGS.mode == "learn_basis":
                sv.restore(session, tf.train.latest_checkpoint(os.path.join(FLAGS.save_path, "pretrain")))

                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin learning embedding basis!")
                learn_sparse_embedding(session, mtrain, verbose=True)
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish learning embedding basis!")

                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin learning softmax basis!")
                learn_sparse_softmax(session, mtrain, verbose=True)
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish learning softmax basis!")
                sys.exit(0)

            elif FLAGS.mode == "finetune":
                # Restore pre-trained model
                sv.restore(session, tf.train.latest_checkpoint(os.path.join(FLAGS.save_path, "pretrain")))
                for i in range(config.finetune_epoch // FLAGS.laptop_discount):
                    lr_decay = config.lr_decay ** (i // config.max_epoch)
                    mtrain.assign_lr(session, config.finetune_learning_rate * lr_decay)

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)), file=logfile)
                    train_perplexity = run_epoch(session, mtrain, eval_op=mtrain.train_op, data_feeder=train_feeder,
                                                 verbose=True)

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity), file=logfile)
                    logfile.flush()

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    valid_perplexity = run_epoch(session, mvalid, data_feeder=valid_feeder)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity), file=logfile)
                    logfile.flush()

                    # Save model if FLAGS.mode == "pretrain" or "finetune"
                    if FLAGS.save_path:
                        print("Saving model to %s." % FLAGS.save_path, file=logfile)
                        step = mtrain.get_global_step(session)
                        finetune_save_path = os.path.join(FLAGS.save_path, "finetune-" + FLAGS.model_config)
                        if not os.path.isdir(finetune_save_path):
                            os.mkdir(finetune_save_path)
                        model_save_path = os.path.join(finetune_save_path, FLAGS.model_name)
                        sv.save(session, model_save_path, global_step=step)

                    # Export sparse graph at every iteration
                    print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin exporting graph!")
                    export_graph(session)  # Export dense graph
                    print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish exporting graph!")

                # Evaluate on test data for {"pretrain", "finetune",} phase
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin test epoch!")
                sys.stdout.flush()
                print("=" * 30 + FLAGS.mode + "=" * 30, file=logfile)
                test_perplexity = run_evaluate_epoch(session, mtest, logfile, word_dict=data_utility.id2token_out, data_feeder=test_feeder)
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish test epoch!")
                print("Test Perplexity: %.3f" % test_perplexity, file=logfile)
                print("Test Perplexity: %.3f" % test_perplexity)  # print to stdout
                logfile.close()

if __name__ == "__main__":
    tf.app.run()
