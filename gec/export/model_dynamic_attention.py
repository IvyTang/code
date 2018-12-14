from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import config

FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type


class Model(object):
    """Static PTB model. Modified from old saniti-checked version of dynamic model.
    """

    def __init__(self, is_training, config, initializer):

        self.lm_embedding_size = config.word_embedding_size
        self.lm_hidden_size = config.hidden_size
        self.lm_vocab_size_in = config.vocab_size_in
        self.lm_vocab_size_out = config.vocab_size_out
        self.lm_input_data = tf.placeholder(dtype=index_data_type(), shape=[None, None],
                                            name="lm_batched_input_word_ids")
        self.lemma_input_data = tf.placeholder(dtype=index_data_type(), shape=[None, None],
                                               name="batched_lemma_input_word_ids")
        self.lm_sequence_length = tf.placeholder(dtype=index_data_type(), shape=[None],
                                                 name="lm_batched_input_word_sequence")
        self.kc_top_k = tf.placeholder(dtype=index_data_type(), shape=[None], name="kc_top_k")
        input_shape = tf.shape(self.lm_input_data)
        (lm_batch_size, time_steps) = tf.unstack(input_shape, 2)

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.lm_hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell_fw = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        cell_bw = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        initial_state_fw = cell_fw.zero_state(lm_batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(lm_batch_size, tf.float32)

        self.attention_size = config.word_embedding_size

        with tf.variable_scope("WordModel", reuse=False, initializer=initializer):
            with tf.variable_scope("Lm"):
                with tf.variable_scope("Embedding"):
                    self.lm_embedding = tf.get_variable("embedding", [self.lm_vocab_size_in, self.lm_embedding_size], dtype=data_type())
                    inputs = tf.nn.embedding_lookup(self.lm_embedding, self.lm_input_data)
                    lemma_inputs = tf.nn.embedding_lookup(self.lm_embedding, self.lemma_input_data)
                    self.lemma_embedding = tf.reshape(lemma_inputs, [-1, self.lm_embedding_size])
                    embedding_to_rnn = tf.get_variable("embedding_to_rnn",
                                                      [self.lm_embedding_size, self.lm_hidden_size],
                                                      dtype=data_type())
                    inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.lm_embedding_size]),
                                              embedding_to_rnn),
                                    shape=[lm_batch_size, -1, self.lm_hidden_size])

                    if is_training and config.keep_prob < 1:
                        inputs = tf.nn.dropout(inputs, config.keep_prob)

                with tf.variable_scope("BiRNN"):

                    with tf.variable_scope('fw'):
                        inputs = tf.transpose(inputs, [1, 0, 2])
                        state_ta = tf.TensorArray(size=time_steps, dtype=tf.float32)
                        input_ta = tf.TensorArray(size=time_steps, dtype=tf.float32)
                        input_ta = input_ta.unstack(inputs)

                        with tf.variable_scope('Attention_layer'):
                            f_w_attention = tf.get_variable('W_Attention', [self.lm_hidden_size, self.attention_size],
                                                          dtype=data_type())

                        def body(time, state_ta_t, state):
                            xt = input_ta.read(time)
                            new_output, new_state = cell_fw(xt, state)
                            state_ta_t = state_ta_t.write(time, new_output)
                            return (time + 1, state_ta_t, new_state)

                        def condition(time, states, state):
                            return time < time_steps

                        time = 0
                        state_fw = initial_state_fw

                        time_final, state_ta_final, state_final = tf.while_loop(
                            cond=condition,
                            body=body,
                            loop_vars=(time, state_ta, state_fw))

                        states_fw = state_ta_final.stack()

                        att_state_ta = tf.TensorArray(size=time_steps, dtype=tf.float32)

                        def body(time, att_state_t):
                            xt = states_fw[:(time + 1), :, :]
                            lemma_input = lemma_inputs[:, time, :]  # (B, D)
                            lemma_input = tf.tile(tf.expand_dims(lemma_input, 1),
                                                  [1, time + 1, 1])  # (B, t, embedding_size)
                            cur_xt = states_fw[time, :, :]
                            f_rnn_output_t = tf.transpose(xt, perm=[1, 0, 2])
                            vu = tf.reduce_sum(tf.tensordot(f_rnn_output_t, f_w_attention, 1) * lemma_input,
                                               2)  # (B, t)
                            alphas = tf.nn.softmax(vu, name='alphas')  # (B, t)
                            attention_output = tf.reduce_sum(f_rnn_output_t * tf.expand_dims(alphas, -1), 1)  # (B, D)

                            attention_output_concat = tf.concat([attention_output, cur_xt], axis=1)
                            att_state_t = att_state_t.write(time, attention_output_concat)
                            return (time + 1, att_state_t)

                        def condition(time, att_states):
                            return time < time_steps

                        time = 0
                        time_final, att_state_ta_final = tf.while_loop(
                            cond=condition,
                            body=body,
                            loop_vars=(time, att_state_ta))

                        fw_att_outputs = att_state_ta_final.stack()

                    # backward direction
                    lemma_inputs = tf.reverse_sequence(
                        lemma_inputs, tf.subtract(self.lm_sequence_length, 2 * tf.ones([lm_batch_size], dtype=tf.int32)),
                        seq_axis=1, batch_axis=0)

                    with tf.variable_scope('bw'):
                        inputs = tf.transpose(inputs, [1, 0, 2])
                        # inputs_t: [batch_size, time_step, embedding_size]
                        input_bw = tf.TensorArray(size=lm_batch_size, dtype=tf.float32)
                        input_ta = tf.TensorArray(size=lm_batch_size, dtype=tf.float32)
                        input_ta = input_ta.unstack(inputs)

                        with tf.variable_scope('Attention_layer'):
                            b_w_attention = tf.get_variable('W_Attention', [self.lm_hidden_size, self.attention_size],
                                                          dtype=data_type())

                        def body(time, input_bw, input_ta):
                            xt = input_ta.read(time)
                            xt = tf.concat(
                                [tf.reverse(xt[:self.lm_sequence_length[time], :], [0]),
                                 xt[self.lm_sequence_length[time]:, :]], axis=0)

                            input_bw = input_bw.write(time, xt)
                            return (time + 1, input_bw, input_ta)

                        def condition(time, input_bw, input_ta):
                            return time < lm_batch_size

                        time = 0

                        time_final, inputs_bw_final, input_ta_final = tf.while_loop(
                            cond=condition,
                            body=body,
                            loop_vars=(time, input_bw, input_ta))

                        inputs = inputs_bw_final.stack()
                        inputs = tf.transpose(inputs, [1, 0, 2])
                        # inputs: [time_steps, batch_size, embedding_size]

                        state_ta = tf.TensorArray(size=time_steps, dtype=tf.float32)
                        input_ta = tf.TensorArray(size=time_steps, dtype=tf.float32)
                        input_ta = input_ta.unstack(inputs)

                        def body(time, state_ta_t, state):
                            xt = input_ta.read(time)
                            new_output, new_state = cell_bw(xt, state)
                            # output_ta_t = output_ta_t.write(time, new_output)
                            state_ta_t = state_ta_t.write(time, new_output)
                            return (time + 1, state_ta_t, new_state)

                        def condition(time, states, state):
                            return time < time_steps

                        time = 0
                        state_bw = initial_state_bw

                        time_final, state_ta_final, state_final = tf.while_loop(
                            cond=condition,
                            body=body,
                            loop_vars=(time, state_ta, state_bw))

                        states_bw = state_ta_final.stack()

                        att_state_ta = tf.TensorArray(size=time_steps, dtype=tf.float32)

                        def body(time, att_state_t):
                            xt = states_bw[:(time + 1), :, :]
                            lemma_input = lemma_inputs[:, time, :]  # (B, D)
                            lemma_input = tf.tile(tf.expand_dims(lemma_input, 1),
                                                  [1, time + 1, 1])  # (B, t, embedding_size)
                            cur_xt = states_bw[time, :, :]
                            f_rnn_output_t = tf.transpose(xt, perm=[1, 0, 2])
                            vu = tf.reduce_sum(tf.tensordot(f_rnn_output_t, b_w_attention, 1) * lemma_input,
                                               2)  # (B, t)
                            alphas = tf.nn.softmax(vu, name='alphas')  # (B, t)
                            attention_output = tf.reduce_sum(f_rnn_output_t * tf.expand_dims(alphas, -1), 1)  # (B, D)
                            attention_output_concat = tf.concat([attention_output, cur_xt], axis=1)
                            att_state_t = att_state_t.write(time, attention_output_concat)
                            return (time + 1, att_state_t)

                        def condition(time, att_states):
                            return time < time_steps

                        time = 0
                        time_final, att_state_ta_final = tf.while_loop(
                            cond=condition,
                            body=body,
                            loop_vars=(time, att_state_ta))
                        bw_att_outputs = att_state_ta_final.stack()

                    states_bw = tf.transpose(states_bw, perm=[1, 0, 2])
                    # states_bw.shape: [batch_size, time_steps, hidden_size]

                    states_ta = tf.TensorArray(size=lm_batch_size, dtype=tf.float32)
                    states_bw_ta = tf.TensorArray(size=lm_batch_size, dtype=tf.float32)
                    states_ta = states_ta.unstack(states_bw)

                    def body(time, states_bw_ta, states_ta):
                        xt = states_ta.read(time)
                        xt = tf.concat(
                            [tf.reverse(xt[:(self.lm_sequence_length[time]-2), :], [0]),
                             xt[(self.lm_sequence_length[time]-2):(time_steps-2), :]], axis=0)

                        states_bw_ta = states_bw_ta.write(time, xt)
                        return (time + 1, states_bw_ta, states_ta)

                    def condition(time, states_bw_ta, states_ta):
                        return time < lm_batch_size

                    time = 0

                    time_final, states_bw_ta_final, states_ta_final = tf.while_loop(
                        cond=condition,
                        body=body,
                        loop_vars=(time, states_bw_ta, states_ta))

                    states_bw = states_bw_ta_final.stack()

                    states_fw = tf.transpose(states_fw, perm=[1, 0, 2])
                    states_ta = tf.TensorArray(size=lm_batch_size, dtype=tf.float32)
                    states_fw_ta = tf.TensorArray(size=lm_batch_size, dtype=tf.float32)
                    states_ta = states_ta.unstack(states_fw)

                    def body(time, states_fw_ta, states_ta):
                        xt = states_ta.read(time)
                        xt = xt[:(time_steps-2), :]

                        states_fw_ta = states_fw_ta.write(time, xt)
                        return (time + 1, states_fw_ta, states_ta)

                    def condition(time, states_fw_ta, states_ta):
                        return time < lm_batch_size

                    time = 0

                    time_final, states_fw_ta_final, states_ta_final = tf.while_loop(
                        cond=condition,
                        body=body,
                        loop_vars=(time, states_fw_ta, states_ta))

                    states_fw = states_fw_ta_final.stack()

                    fw_att_outputs = tf.transpose(fw_att_outputs, perm=[1, 0, 2])
                    bw_att_outputs = tf.transpose(bw_att_outputs, perm=[1, 0, 2])

                    fw_att_outputs = tf.slice(fw_att_outputs, [0, 0, 0],
                                              [lm_batch_size, time_steps - 2, self.lm_hidden_size * 2])
                    bw_att_outputs = tf.slice(bw_att_outputs, [0, 0, 0],
                                              [lm_batch_size, time_steps - 2, self.lm_hidden_size * 2])

                    final_state = tf.concat([states_fw, states_bw], 2)
                    final_state = tf.reshape(final_state, [-1, self.lm_hidden_size * 2])

                    att_output = tf.concat([fw_att_outputs, bw_att_outputs], 2)
                    att_output = tf.reshape(att_output, [-1, self.lm_hidden_size * 4])

                with tf.variable_scope("Softmax"):
                    rnn_output_to_final_output = tf.get_variable("rnn_output_to_final_output",
                                                                 [self.lm_hidden_size * 2, self.lm_embedding_size],
                                                                 dtype=data_type())
                    # self.lm_softmax_w = tf.get_variable("softmax_w", [self.lm_embedding_size, self.lm_vocab_size_out],
                    #                                   dtype=data_type())
                    # lm_softmax_b = tf.get_variable("softmax_b", [self.lm_vocab_size_out], dtype=data_type())

                    self.word_output = tf.matmul(final_state, rnn_output_to_final_output)
                    # word_logits = tf.matmul(self.word_output, self.lm_softmax_w) + lm_softmax_b
            with tf.variable_scope("LemmaSoftmax"):
                att_output_to_final_output = tf.get_variable("att_output_to_final_output",
                                                             [self.lm_hidden_size * 4, self.lm_embedding_size],
                                                             dtype=data_type())
                W_matrix = tf.get_variable(
                    "W_matrix",
                    shape=[self.lm_embedding_size, self.lm_embedding_size],
                    dtype=tf.float32)
                U_matrix = tf.get_variable(
                    "U_matrix",
                    shape=[self.lm_embedding_size, self.lm_embedding_size],
                    dtype=tf.float32)

                self.att_final_output = tf.matmul(tf.matmul(att_output, att_output_to_final_output), W_matrix) \
                                        + tf.matmul(self.lemma_embedding, U_matrix)

                lemma_W = tf.get_variable(
                    "W",
                    shape=[self.lm_embedding_size, self.lm_vocab_size_out], dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                lemma_b = tf.get_variable("b", [self.lm_vocab_size_out], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.1))
                att_logits = tf.nn.xw_plus_b(self.att_final_output, lemma_W, lemma_b,
                                             name="scores")
            self.att_probabilities = tf.nn.softmax(att_logits, name="att_probabilities")
            self.lemma_top_k_probs, self.lemma_top_k_prediction = tf.nn.top_k(
                self.att_probabilities, self.kc_top_k[0], name="att_top_k_prediction")
            self.lemma_top_k_prediction = tf.reshape(self.lemma_top_k_prediction,
                                                     [lm_batch_size, time_steps, self.kc_top_k[0]])
            self.lemma_top_k_probs = tf.reshape(self.lemma_top_k_probs,
                                                [lm_batch_size, time_steps, self.kc_top_k[0]])

        sequence_length = 20
        num_classes = config.vocab_size_out
        vocab_size = config.vocab_size_letter
        embedding_size = config.letter_embedding_size
        filter_sizes = [3, 4, 5]
        num_filters = 128

        self.input_x = tf.placeholder(tf.int32, [None, None, sequence_length], name="input_x")

        input_shape = tf.shape(self.input_x)
        (lm_batch_size, max_sentence_length, kc_time_steps) = tf.unstack(input_shape, 3)
        kc_input_data = tf.reshape(self.input_x, [-1, sequence_length])
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, dtype=tf.float32)

        with tf.variable_scope("LetterModel"):
            # Embedding layer
            with tf.variable_scope("CNN"):
                with tf.variable_scope("embedding"):
                    self.W = tf.get_variable("W",
                                        [vocab_size, embedding_size], dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0)
                                        )
                    self.embedded_chars = tf.nn.embedding_lookup(self.W, kc_input_data)
                    self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

                # Create a convolution + maxpool layer for each filter size
                pooled_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.variable_scope("conv-maxpool-%s" % i):
                        # Convolution Layer
                        filter_shape = [filter_size, embedding_size, 1, num_filters]
                        W = tf.get_variable("W", filter_shape, dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                        b = tf.get_variable("b", [num_filters], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.1))
                        conv = tf.nn.conv2d(
                            self.embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        # Maxpooling over the outputs
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                        pooled_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = num_filters * len(filter_sizes)
                self.h_pool = tf.concat(pooled_outputs, 3)
                self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

                # Add dropout
                if is_training:
                    self.h_pool_flat = tf.nn.dropout(self.h_pool_flat, 0.5)

                # Final (unnormalized) scores and predictions
                with tf.variable_scope("output"):
                    pool_flat_to_embedding_matrix = tf.get_variable(
                        "pool_flat_to_embedding_matrix",
                        shape=[num_filters_total, embedding_size],
                        dtype=tf.float32)
                    output_trans_matrix = tf.get_variable("w2n", shape=[embedding_size + config.word_embedding_size,
                                                                        embedding_size], dtype=tf.float32)
                    W = tf.get_variable(
                        "W",
                        shape=[embedding_size, num_classes], dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable("b", [num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    self.output = tf.matmul(self.h_pool_flat, pool_flat_to_embedding_matrix)
                    self.output_concated_lm = tf.concat([self.output, self.word_output], axis=1)
                    # self.scores = tf.nn.xw_plus_b(self.output_concated_lm, W, b,
                    #                               name="scores")
                    self.scores = tf.nn.xw_plus_b(tf.matmul(self.output_concated_lm, output_trans_matrix), W, b,
                                                  name="scores")
                    self.probs = tf.nn.softmax(self.scores, name="probabilities")
                    self.top_k_probs, self.top_k_prediction = tf.nn.top_k(self.probs, self.kc_top_k[0],
                                                                          name="kc_top_k_prediction")
                    self.top_k_prediction = tf.reshape(self.top_k_prediction,
                                                          [lm_batch_size, max_sentence_length, self.kc_top_k[0]])
                    self.top_k_probs = tf.reshape(self.top_k_probs,
                                                       [lm_batch_size, max_sentence_length, self.kc_top_k[0]])







