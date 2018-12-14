#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
from config import Config
import tensorflow as tf
from model_dynamic_attention import Model


def save_serving_model(sess, export_path, model):
    # 版本号和目录结构
    model_version = 1
    version_export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(model_version)))
    print('Exporting trained model to', version_export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(version_export_path)

    # Build the signature_def_map.
    # inputs & outputs
    lm_tensor_input = tf.saved_model.utils.build_tensor_info(model.lm_input_data)
    lemma_tensor_input = tf.saved_model.utils.build_tensor_info(model.lemma_input_data)
    lm_tensor_length = tf.saved_model.utils.build_tensor_info(model.lm_sequence_length)
    kc_tensor_input = tf.saved_model.utils.build_tensor_info(model.input_x)
    top_k = tf.saved_model.utils.build_tensor_info(model.kc_top_k)

    tensor_output_values = tf.saved_model.utils.build_tensor_info(model.top_k_prediction)
    tensor_output_probs = tf.saved_model.utils.build_tensor_info(model.top_k_probs)
    tensor_output_lemma_values = tf.saved_model.utils.build_tensor_info(model.lemma_top_k_prediction)
    tensor_output_lemma_probs = tf.saved_model.utils.build_tensor_info(model.lemma_top_k_probs)

    # signature_def
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            # inputs name: input
            inputs={'lm_input': lm_tensor_input,
                    'lemma_input': lemma_tensor_input,
                    'lm_length': lm_tensor_length,
                    'kc_input': kc_tensor_input,
                    'top_k': top_k},
            # outputs name: output_values & output indices
            outputs={'output_values': tensor_output_values, 'output_probs': tensor_output_probs,
                     'lemma_output_values': tensor_output_lemma_values, 'lemma_output_probs': tensor_output_lemma_probs},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    # signature name: 'predict_words'
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_words': prediction_signature
        })
    builder.save()
    print('Done exporting!')
    return


if __name__ == "__main__":
    # how to run, 以 load 特定的 model 为例，如果训练完之后直接导出，则直接调用:
    args = sys.argv

    model_file = args[1]
    vocab_path = args[2]
    config_name = args[3]
    export_path = args[4]

    config = Config()
    config.get_config(vocab_path, config_name)

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.name_scope("Inference"):

            model_test = Model(is_training=False, config=config, initializer=initializer)

        with tf.Session() as session:
            restore_variables = dict()
            for v in tf.trainable_variables():
                print("restore:", v.name)
                restore_variables[v.name] = v
            saver = tf.train.Saver(restore_variables)
            saver.restore(session, model_file)

            save_serving_model(session, export_path, model_test)
