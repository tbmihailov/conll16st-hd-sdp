#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNNModel


def text_cnn_load_model_and_eval(x_test,
                                 checkpoint_file,
                                 allow_soft_placement,
                                 log_device_placement,
                                 embeddings):

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batch_size = 50
            batches = data_helpers.batch_iter(x_test, batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            # Load embeddings placeholder
            embedding_size = embeddings.shape[1]
            embeddings_number = embeddings.shape[0]
            print 'embedding_size:%s, embeddings_number:%s' % (embedding_size, embeddings_number)
            # with tf.name_scope("embedding"):
            #     embeddings_placeholder = tf.placeholder(tf.float32, shape=[embeddings_number, embedding_size])
            embeddings_placeholder = graph.get_operation_by_name("embedding/Placeholder").outputs[0]

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0,
                                                           embeddings_placeholder: embeddings})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    return all_predictions

def text_cnn_load_model_and_eval_v2(x_test_s1,
                                    x_test_s2,
                  checkpoint_file,
                  allow_soft_placement,
                  log_device_placement,
                  embeddings):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x_s1 = graph.get_operation_by_name("input_x_s1").outputs[0]
            input_x_s2 = graph.get_operation_by_name("input_x_s2").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batch_size = 50
            batches = data_helpers.batch_iter(list(zip(x_test_s1, x_test_s2)), batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            # Load embeddings placeholder
            embedding_size = embeddings.shape[1]
            embeddings_number = embeddings.shape[0]
            print 'embedding_size:%s, embeddings_number:%s' % (embedding_size, embeddings_number)
            # with tf.name_scope("embedding"):
            #     embeddings_placeholder = tf.placeholder(tf.float32, shape=[embeddings_number, embedding_size])
            embeddings_placeholder = graph.get_operation_by_name("embedding/Placeholder").outputs[0]

            for batch in batches:
                x_test_batch_s1, x_test_batch_s2 = zip(*batch)
                batch_predictions = sess.run(predictions, {input_x_s1: x_test_batch_s1,
                                                           input_x_s2: x_test_batch_s2,
                                                           dropout_keep_prob: 1.0,
                                                           embeddings_placeholder: embeddings})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    return all_predictions

def text_cnn_load_model_and_eval_v4(#x_test_s1,
                                    #    x_test_s2,
                                    loaded_cross_batch_iter,
                                        checkpoint_file,
                                        allow_soft_placement,
                                        log_device_placement,
                                        embeddings):


    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            #input_x_s1 = graph.get_operation_by_name("input_x_s1").outputs[0]
            #input_x_s2 = graph.get_operation_by_name("input_x_s2").outputs[0]

            input_x_s1s2_cross = graph.get_operation_by_name("input_x_s1s2_cross").outputs[0]
            #cnn.input_y: y_batch,

            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            all_predictions = []
            for loaded_cross_batch, batch_i, items_per_batch in loaded_cross_batch_iter:
                # Generate batches for one epoch
                batch_size = 50
                #batches = data_helpers.batch_iter(list(zip(x_test_s1, x_test_s2)), batch_size, 1, shuffle=False)
                batches = data_helpers.batch_iter(loaded_cross_batch, batch_size, 1, shuffle=False)

                # Collect the predictions here


                # Load embeddings placeholder
                embedding_size = embeddings.shape[1]
                embeddings_number = embeddings.shape[0]
                print 'embedding_size:%s, embeddings_number:%s' % (embedding_size, embeddings_number)
                # with tf.name_scope("embedding"):
                #     embeddings_placeholder = tf.placeholder(tf.float32, shape=[embeddings_number, embedding_size])
                embeddings_placeholder = graph.get_operation_by_name("embedding/Placeholder").outputs[0]

                for batch in batches:
                    #x_test_batch_s1, x_test_batch_s2 = zip(*batch)
                    x_s1s2_cross = batch
                    batch_predictions = sess.run(predictions, {#input_x_s1: x_test_batch_s1,
                                                               #input_x_s2: x_test_batch_s2,
                                                               input_x_s1s2_cross: x_s1s2_cross,
                                                               dropout_keep_prob: 1.0,
                                                               embeddings_placeholder: embeddings})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

    return all_predictions

if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Eval Parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Load data. Load your own data here
    print("Loading data...")
    x_test, y_test, vocabulary, vocabulary_inv = data_helpers.load_data()
    y_test = np.argmax(y_test, axis=1)
    print("Vocabulary size: {:d}".format(len(vocabulary)))
    print("Test set size {:d}".format(len(y_test)))

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    predictions_y = text_cnn_load_model_and_eval(x_test=x_test,
                                  checkpoint_file=checkpoint_file,
                                  allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    # Print accuracy
    correct_predictions = float(sum(predictions_y == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
