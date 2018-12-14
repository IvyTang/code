from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import config

FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type


class WordModel(object):
    # Below is the language model.

    def __init__(self, is_training, cfg):
        self.batch_size = cfg.batch_size
        self.num_steps = cfg.num_steps
        self.embedding_size = cfg.word_embedding_size
        self.hidden_size = cfg.word_hidden_size
        self.vocab_size_in = cfg.vocab_size_in
        self.vocab_size_out = cfg.vocab_size_out
        self.vocab_size_phrase = cfg.vocab_size_phrase
        self.vocab_size_emoji = cfg.vocab_size_emoji

        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                         name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                          name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_word_masks")
        self.sequence_length = tf.placeholder_with_default(
            input=tf.fill(dims=[self.batch_size], value=self.num_steps),
            shape=[self.batch_size], name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")

        self.target_phrase_p = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                              name="batched_output_phrase_p_ids")
        self.target_phrase_p_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                                    name="batched_output_phrase_p_masks")
        self.target_phrase_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                                 name="batched_output_phrase_ids")
        self.target_phrase_data_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                                       name="batched_output_phrase_masks")
        self.target_phrase_logits_masks = tf.placeholder_with_default(
            tf.ones([self.batch_size * self.num_steps, self.vocab_size_phrase], dtype=data_type()),
            [self.batch_size * self.num_steps, self.vocab_size_phrase], name="batched_output_phrase_logits_masks")
        self.target_emoji_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                                name="batched_output_emoji_ids")
        self.target_emoji_mask = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                                name="batched_output_emoji_masks")

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and cfg.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=cfg.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(cfg.num_layers)], state_is_tuple=True)

        lstm_state_as_tensor_shape = [cfg.num_layers, 2, cfg.batch_size, cfg.word_hidden_size]
        
        self._initial_state = tf.placeholder_with_default(tf.zeros(lstm_state_as_tensor_shape, dtype=data_type()),
                                                          lstm_state_as_tensor_shape, name="state")

        unstack_state = tf.unstack(self._initial_state, axis=0)
        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1])
             for idx in range(cfg.num_layers)]
        )

        with tf.variable_scope("Lm"):
            with tf.variable_scope("Embedding"):
                self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size],
                                                  dtype=data_type())
                inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)

                # inputs is of shape [batch_size, num_steps, word_embedding_size]

                embedding_to_rnn = tf.get_variable("embedding_to_rnn", [self.embedding_size, self.hidden_size],
                                                   dtype=data_type())
                inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]), embedding_to_rnn),
                                    shape=[self.batch_size, -1, self.hidden_size])

                # Now inputs is of shape [batch_size, num_steps, word_hidden_size]

                if is_training and cfg.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, cfg.keep_prob)

            with tf.variable_scope("RNN"):
                outputs = list()
                states = list()
                state = tuple_state

                for timestep in range(self.num_steps):

                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output, state) = cell(inputs[:, timestep, :], state)
                    outputs.append(output)
                    states.append(state)

                rnn_output = tf.transpose(outputs, perm=[1, 0, 2])
                # rnn_output is a Tensor of shape [batch_size, num_steps, word_hidden_size]

                rnn_output = tf.reshape(rnn_output, [-1, self.hidden_size])
                # Now rnn_output is a Tensor of shape [batch_size * num_steps, word_hidden_size]

                states = tf.transpose(states, perm=[3, 1, 2, 0, 4])
                # states is a Tensor of shape [batch_size, num_layers, 2, num_steps, word_hidden_size]

                unstack_states = tf.unstack(states, axis=0)
                rnn_state = tf.concat(unstack_states, axis=2)
                # Now rnn_state is a Tensor of shape [num_layers, 2, batch_size * num_steps, word_hidden_size]

            with tf.variable_scope("Softmax"):
                rnn_output_to_final_output = tf.get_variable("rnn_output_to_final_output",
                                                             [self.hidden_size, self.embedding_size],
                                                             dtype=data_type())
                self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                                  dtype=data_type())
                softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())

        with tf.variable_scope("PhraseProb"):
            self._softmax_phrase_p_w = tf.get_variable("softmax_phrase_p_w", [self.embedding_size, 2],
                                                       dtype=data_type())
            softmax_phrase_p_b = tf.get_variable("softmax_phrase_p_b", [2], dtype=data_type())
        with tf.variable_scope("Phrase"):
            self._softmax_phrase_w = tf.get_variable("softmax_phrase_w",
                                                     [self.embedding_size, self.vocab_size_phrase],
                                                     dtype=data_type())
            softmax_phrase_b = tf.get_variable("softmax_phrase_b", [self.vocab_size_phrase], dtype=data_type())

        self._logits_phrase_p = logits_phrase_p = tf.matmul(tf.matmul(rnn_output, rnn_output_to_final_output),
                                                            self._softmax_phrase_p_w) + softmax_phrase_p_b
        self._phrase_p_probabilities = tf.nn.softmax(logits_phrase_p, name="phrase_p_probabilities")
        # phrase_p_probabilities.shape = [batch_size * num_steps, 2]

        _, phrase_p_prediction = tf.nn.top_k(logits_phrase_p, 2, name="phrase_p_prediction")

        logits_phrase = (tf.matmul(tf.matmul(rnn_output, rnn_output_to_final_output),
                                   self._softmax_phrase_w) + softmax_phrase_b) * self.target_phrase_logits_masks
        self._logits_phrase = tf.identity(logits_phrase, name="logits_phrase")
        self._phrase_probabilities = tf.nn.softmax(logits_phrase, name="phrase_probabilities")
        # phrase_probabilities.shape = [batch_size * num_steps, vocab_size_out]

        _, self._phrase_prediction = tf.nn.top_k(logits_phrase, self.top_k, name="phrase_top_k_prediction")

        loss_phrase_p = tf.contrib.seq2seq.sequence_loss(
            tf.reshape(logits_phrase_p, [self.batch_size, self.num_steps, 2]),
            self.target_phrase_p, self.target_phrase_p_masks)

        loss_phrase = tf.contrib.seq2seq.sequence_loss(
            tf.reshape(logits_phrase, [self.batch_size, self.num_steps, self.vocab_size_phrase]),
            self.target_phrase_data, self.target_phrase_data_masks)

        self._phrase_p_cost = loss_phrase_p

        self._phrase_cost = loss_phrase

        with tf.variable_scope("Emoji"):
            self._softmax_emoji_w = tf.get_variable("softmax_emoji_w", [self.embedding_size, self.vocab_size_emoji],
                                                    dtype=data_type())
            softmax_emoji_b = tf.get_variable("softmax_emoji_b", [self.vocab_size_emoji], dtype=data_type())
        self._logits_emoji = logits_emoji = tf.matmul(tf.matmul(rnn_output, rnn_output_to_final_output),
                                                      self._softmax_emoji_w) + softmax_emoji_b
        self._emoji_probabilities = tf.nn.softmax(logits_emoji, name="emoji_probabilities")
        # emoji_probabilities.shape = [batch_size * num_steps, vocab_size_emoji]

        _, self._emoji_prediction = tf.nn.top_k(logits_emoji, self.top_k, name="emoji_top_k_prediction")
        loss_emoji = tf.contrib.seq2seq.sequence_loss(
            tf.reshape(logits_emoji, [self.batch_size, self.num_steps, self.vocab_size_emoji]),
            self.target_emoji_data, self.target_emoji_mask)

        self._emoji_cost = loss_emoji

        logits = tf.matmul(tf.matmul(rnn_output, rnn_output_to_final_output), self._softmax_w) + softmax_b

        probabilities = tf.nn.softmax(logits, name="probabilities")
        # probabilities.shape = [batch_size * num_steps, vocab_size_out]

        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        loss_word = tf.contrib.seq2seq.sequence_loss(
            tf.reshape(logits, [self.batch_size, self.num_steps, self.vocab_size_out]),
            self.target_data, self.output_masks)

        self._cost = loss_word

        self._final_state = tf.identity(state, "state_out")
        self._rnn_state = tf.identity(rnn_state, "rnn_state")
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(cfg.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Lm")
        tvars_phrase_p = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/PhraseProb")
        tvars_phrase = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Phrase")

        grads, _ = tf.clip_by_global_norm(tf.gradients(loss_word, tvars), cfg.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        grads_phrase_p, _ = tf.clip_by_global_norm(tf.gradients(loss_phrase_p, tvars_phrase_p), cfg.max_grad_norm)
        optimizer_phrase_p = tf.train.AdamOptimizer(0.001)
        self._train_op_phrase_p = optimizer_phrase_p.apply_gradients(zip(grads_phrase_p, tvars_phrase_p),
                                                                     global_step=self.global_step)

        grads_phrase, _ = tf.clip_by_global_norm(tf.gradients(loss_phrase, tvars_phrase), cfg.max_grad_norm)
        optimizer_phrase = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op_phrase = optimizer_phrase.apply_gradients(zip(grads_phrase, tvars_phrase),
                                                                 global_step=self.global_step)

        tvars_emoji = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Emoji")
        grads_emoji, _ = tf.clip_by_global_norm(tf.gradients(loss_emoji, tvars_emoji), cfg.max_grad_norm)
        optimizer_emoji = tf.train.AdamOptimizer(0.001)
        self._train_op_emoji = optimizer_emoji.apply_gradients(zip(grads_emoji, tvars_emoji),
                                                               global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

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
        return [self._cost, self._phrase_p_cost, self._phrase_cost, self._emoji_cost]

    @property
    def embedding(self):
        return self._embedding

    @property
    def final_state(self):
        return self._final_state

    @property
    def rnn_state(self):
        return self._rnn_state

    @property
    def lr(self):
        return self._lr

    @property
    def logits(self):
        return [self._logits, self._logits_phrase_p, self._logits_phrase, self._logits_emoji]

    @property
    def probalities(self):
        return [self._probabilities, self._phrase_p_probabilities,
                self._phrase_probabilities, self._emoji_probabilities]

    @property
    def top_k_prediction(self):
        return [self._top_k_prediction, self._phrase_prediction, self._emoji_prediction]

    @property
    def train_op(self):
        return [self._train_op, self._train_op_phrase_p, self._train_op_phrase, self._train_op_emoji]


class LetterModel(object):
    # Below is the letter model.

    def __init__(self, is_training, cfg):
        self.num_steps = cfg.num_steps
        self.batch_size = cfg.batch_size * cfg.num_steps
        # the batch_size of letter model equals to batch_size * num_steps

        self.max_word_length = cfg.max_word_length
        self.embedding_size = cfg.letter_embedding_size
        self.hidden_size = cfg.letter_hidden_size
        self.vocab_size_in = cfg.vocab_size_letter
        self.vocab_size_out = cfg.vocab_size_out

        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                         name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                          name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_word_masks")
        self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size],
                                                           value=self.max_word_length),
                                                           shape=[self.batch_size],
                                                           name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and cfg.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=cfg.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(cfg.num_layers)], state_is_tuple=True)

        lm_state_as_tensor_shape = [cfg.num_layers, 2, self.batch_size, cfg.word_hidden_size]
        letter_state_as_tensor_shape = [cfg.num_layers, 2, self.batch_size, cfg.letter_hidden_size]

        self.lm_state_in = tf.placeholder_with_default(tf.zeros(lm_state_as_tensor_shape, dtype=data_type()),
                                                       lm_state_as_tensor_shape, name="lm_state_in")
        # lm_state_in is of shape [num_layers, 2, batch_size * num_steps, word_hidden_size]

        with tf.variable_scope("StateMatrix"):

            lm_state_to_letter_state = tf.get_variable("lm_state_to_letter_state",
                                                       [cfg.word_hidden_size, cfg.letter_hidden_size],
                                                       dtype=data_type())

        if cfg.word_hidden_size != cfg.letter_hidden_size:
            self._initial_state = tf.placeholder_with_default(
                                tf.reshape(tf.matmul(tf.reshape(self.lm_state_in, [-1, cfg.word_hidden_size]),
                                                     lm_state_to_letter_state), letter_state_as_tensor_shape),
                                                     letter_state_as_tensor_shape, name="state")
        else:
            self._initial_state = tf.placeholder_with_default(
                self.lm_state_in, letter_state_as_tensor_shape, name="state")

        # initial_state is of shape [num_layers, 2, batch_size * num_steps, letter_hidden_size]

        unstack_state = tf.unstack(self._initial_state, axis=0)
        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1])
             for idx in range(cfg.num_layers)]
        )

        with tf.variable_scope("Embedding"):

            self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size], dtype=data_type())

            inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)
            # inputs is of shape [batch_size * num_steps, max_word_length, letter_embedding_size]

            embedding_to_rnn = tf.get_variable("embedding_to_rnn",
                                               [self.embedding_size, self.hidden_size],
                                               dtype=data_type())
            inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]), embedding_to_rnn),
                                shape=[self.batch_size, -1, self.hidden_size])
            # now inputs is of shape [batch_size * num_steps, max_word_length, letter_hidden_size]

            if is_training and cfg.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, cfg.keep_prob)

        with tf.variable_scope("RNN"):
            outputs, state_out = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.sequence_length,
                                                   initial_state=tuple_state)
            # outputs is a Tensor of shape [batch_size * num_steps, max_word_length, letter_hidden_size]
            # state_out is a tuple of tuple of Tensor: state_out = ((c1, h1), (c2, h2), ..., (cl, hl))

        output = tf.reshape(outputs, [-1, self.hidden_size])
        # Now output is a Tensor of shape [batch_size * num_steps * max_word_length, letter_hidden_size]
        with tf.variable_scope("Softmax"):
            rnn_output_to_final_output = tf.get_variable("rnn_output_to_final_output",
                                                         [self.hidden_size, self.embedding_size],
                                                         dtype=data_type())
            self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                              dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())

        logits = tf.matmul(tf.matmul(output, rnn_output_to_final_output),
                           self._softmax_w) + softmax_b

        probabilities = tf.nn.softmax(logits, name="probabilities")
        # probabilities.shape = [batch_size * num_steps * max_word_length, vocab_size_out]
        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        loss = tf.contrib.seq2seq.sequence_loss(
            tf.reshape(logits, [self.batch_size, self.max_word_length, self.vocab_size_out]),
            self.target_data, self.output_masks)

        self._cost = loss

        self._final_state = tf.identity(state_out, "state_out")
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(cfg.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LetterModel")

        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), cfg.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

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
