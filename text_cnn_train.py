#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNNModel
from text_cnn_cross import TextCNNModel_Cross, TextCNNModel_Cross_Conv


def text_cnn_train_and_save_model(x_train, y_train,
                                  x_dev, y_dev,
                                  out_dir,
                                  allow_soft_placement,
                                  log_device_placement,
                                  embeddings,
                                  vocabulary,
                                  filter_sizes,
                                  num_filters,
                                  l2_reg_lambda,
                                  dropout_keep_prob,
                                  batch_size,
                                  num_epochs,
                                  evaluate_every,
                                  checkpoint_every,
                                  num_classes
                                  ):


    global sess, cnn, global_step, train_op, train_summary_op, train_summary_writer, dev_summary_op
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNNModel(
                sequence_length=x_train.shape[1],
                num_classes=num_classes,
                vocab_size=len(vocabulary),
                embeddings=embeddings,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob,
                    cnn.embeddings_placeholder: embeddings
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.embeddings_placeholder: embeddings
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)


            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), batch_size, num_epochs)

            # Training loop. For each batch...

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                #if current_step % checkpoint_every == 0:
                #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #    print("Saved model checkpoint to {}\n".format(path))
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))




def text_cnn_train_and_save_model_v2(x_train_s1, x_train_s2, y_train,
                                    x_dev_s1, x_dev_s2, y_dev,
                                  out_dir,
                                  allow_soft_placement,
                                  log_device_placement,
                                  embeddings,
                                  vocabulary,
                                  filter_sizes,
                                  num_filters,
                                  l2_reg_lambda,
                                  dropout_keep_prob,
                                  batch_size,
                                  num_epochs,
                                  evaluate_every,
                                  checkpoint_every,
                                  num_classes
                                  ):

    global sess, cnn, global_step, train_op, train_summary_op, train_summary_writer, dev_summary_op
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNNModel_Cross(
                sequence_length=x_train_s1.shape[1],
                num_classes=num_classes,
                vocab_size=len(vocabulary),
                embeddings=embeddings,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())


            def train_step(x_batch_s1, x_batch_s2, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x_s1: x_batch_s1,
                    cnn.input_x_s2: x_batch_s2,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob,
                    cnn.embeddings_placeholder: embeddings
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                # time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch_s1, x_batch_s2, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x_s1: x_batch_s1,
                    cnn.input_x_s2: x_batch_s2,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.embeddings_placeholder: embeddings
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train_s1,x_train_s2, y_train)), batch_size, num_epochs)

            # Training loop. For each batch...

            for batch in batches:
                x_batch_s1,x_batch_s2, y_batch = zip(*batch)
                train_step(x_batch_s1, x_batch_s2, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev_s1, x_dev_s2, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def text_cnn_train_and_save_model_v3(x_train_s1, x_train_s2, x_train_s1s2_cross, y_train,
                                                         x_dev_s1, x_dev_s2,x_dev_s1s2_cross, y_dev,
                                                         out_dir,
                                                         allow_soft_placement,
                                                         log_device_placement,
                                                         embeddings,
                                                         vocabulary,
                                                         filter_sizes,
                                                         num_filters,
                                                         l2_reg_lambda,
                                                         dropout_keep_prob,
                                                         batch_size,
                                                         num_epochs,
                                                         evaluate_every,
                                                         checkpoint_every,
                                                         num_classes
                                                         ):

    global sess, cnn, global_step, train_op, train_summary_op, train_summary_writer, dev_summary_op
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNNModel_Cross_Conv(
                sequence_length=x_train_s1.shape[1],
                num_classes=num_classes,
                vocab_size=len(vocabulary),
                filter_size_cross=3,
                embeddings=embeddings,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())


            def train_step(x_batch_s1, x_batch_s2, x_batch_s1s2_cross, y_batch):
                print "x_batch_s1=%s, x_batch_s2=%s, x_batch_s1s2_cross=%s, y_batch=%s"%(len(x_batch_s1), len(x_batch_s2), len(x_batch_s1s2_cross), len(y_batch))
                print "dropout_keep_prob:%s" % dropout_keep_prob

                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x_s1: x_batch_s1,
                    cnn.input_x_s2: x_batch_s2,
                    cnn.input_x_s1s2_cross: x_batch_s1s2_cross,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob,
                    cnn.embeddings_placeholder: embeddings
                }
                # _, step, summaries, loss, accuracy = sess.run(
                #     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                #     feed_dict)

                _, step, summaries, h_pool_flat, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.h_pool_flat, cnn.accuracy],
                    feed_dict)

                print "h_pool_flat:"
                print h_pool_flat
                # time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch_s1, x_batch_s2, x_batch_s1s2_cross, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x_s1: x_batch_s1,
                    cnn.input_x_s2: x_batch_s2,
                    cnn.input_x_s1s2_cross: x_batch_s1s2_cross,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.embeddings_placeholder: embeddings
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)


            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train_s1, x_train_s2, x_train_s1s2_cross,y_train)), batch_size, num_epochs)

            # Training loop. For each batch...

            for batch in batches:
                x_batch_s1, x_batch_s2, x_batch_s1s2_cross, y_batch = zip(*batch)
                print "y_batch"
                train_step(x_batch_s1, x_batch_s2, x_batch_s1s2_cross, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev_s1, x_dev_s2, x_dev_s1s2_cross, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def text_cnn_train_and_save_model_v4(x_train_s1, x_train_s2, y_train, # , x_train_s1s2_cross
                                     x_dev_s1, x_dev_s2, y_dev, #, x_dev_s1s2_cross
                                     #loaded_cross_batch_iter, this is called in the body
                                     batch_load_cached_conv_settings,
                                     out_dir,
                                     allow_soft_placement,
                                     log_device_placement,
                                     embeddings,
                                     vocabulary,
                                     filter_sizes,
                                     num_filters,
                                     l2_reg_lambda,
                                     dropout_keep_prob,
                                     batch_size,
                                     num_epochs,
                                     evaluate_every,
                                     checkpoint_every,
                                     num_classes):
    global sess, cnn, global_step, train_op, train_summary_op, train_summary_writer, dev_summary_op
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNNModel_Cross_Conv(
                sequence_length=x_train_s1.shape[1],
                num_classes=num_classes,
                vocab_size=len(vocabulary),
                filter_size_cross=3,
                embeddings=embeddings,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())


            def train_step(x_batch_s1, x_batch_s2, x_batch_s1s2_cross, y_batch):
                #print "x_batch_s1=%s, x_batch_s2=%s, x_batch_s1s2_cross=%s, y_batch=%s" % (
                #0 if x_batch_s1==None else (x_batch_s1), 0 if x_batch_s2==None else (x_batch_s2), len(x_batch_s1s2_cross), len(y_batch))
                #print "dropout_keep_prob:%s" % dropout_keep_prob

                """
                A single training step
                """
                feed_dict = {
                    #cnn.input_x_s1: x_batch_s1,
                    #cnn.input_x_s2: x_batch_s2,
                    cnn.input_x_s1s2_cross: x_batch_s1s2_cross,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob,
                    cnn.embeddings_placeholder: embeddings
                }
                # _, step, summaries, loss, accuracy = sess.run(
                #     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                #     feed_dict)

                _, step, summaries, h_pool_flat, accuracy, loss = sess.run(
                    [train_op, global_step, train_summary_op, cnn.h_pool_flat, cnn.accuracy, cnn.loss],
                    feed_dict)

                #print "h_pool_flat:"
                #print h_pool_flat
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch_s1, x_batch_s2, x_batch_s1s2_cross, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    #cnn.input_x_s1: x_batch_s1,
                    #cnn.input_x_s2: x_batch_s2,
                    cnn.input_x_s1s2_cross: x_batch_s1s2_cross,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.embeddings_placeholder: embeddings
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("----------EVAL-----------")
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            x_dev_s1s2_cross = None
            y_dev = None

            l_items_per_batch, ll_all_items_cnt, l_embeddings_len, l_vocab_len = batch_load_cached_conv_settings
            for epoch in range(0, num_epochs):
                print("===Epoch %s of %s===" % (epoch,num_epochs))
                from sup_parser_v6_hierarchy_cnn_cross import DiscourseSenseClassifier_Sup_v6_Hierarchical_CNN_Cross
                # this loads data from disk! slow!!!
                loaded_cross_batch_iter = DiscourseSenseClassifier_Sup_v6_Hierarchical_CNN_Cross.\
                    load_cached_cross_convolution_batch_iter(l_items_per_batch,
                                                             ll_all_items_cnt, l_embeddings_len, l_vocab_len)
                for loaded_cross_batch, batch_i, items_per_batch in loaded_cross_batch_iter:
                    # Generate batches

                    # batches = data_helpers.batch_iter(
                    #     list(zip(x_train_s1, x_train_s2, loaded_cross_batch, y_train)), batch_size, num_epochs)

                    curr_batch_y_train = y_train[batch_i*items_per_batch:batch_i*items_per_batch+loaded_cross_batch.shape[0]]
                    batches = data_helpers.batch_iter(
                        list(zip(loaded_cross_batch, curr_batch_y_train)), batch_size, num_epochs=1)# 1 epoch

                    if x_dev_s1s2_cross==None:
                        x_dev_s1s2_cross = loaded_cross_batch[:100]
                        y_dev = curr_batch_y_train[:100]

                    # Training loop. For each batch...

                    for batch in batches:
                        #x_batch_s1, x_batch_s2, x_batch_s1s2_cross, y_batch = zip(*batch)
                        x_batch_s1s2_cross, y_batch = zip(*batch)

                        #print "y_batch"
                        #train_step(x_batch_s1, x_batch_s2, x_batch_s1s2_cross, y_batch)
                        train_step(None, None, x_batch_s1s2_cross, y_batch)

                        current_step = tf.train.global_step(sess, global_step)
                        if current_step % evaluate_every == 0:
                            print("\nEvaluation:")
                            #dev_step(x_dev_s1, x_dev_s2, x_dev_s1s2_cross, y_dev, writer=dev_summary_writer)
                            dev_step(None, None, x_dev_s1s2_cross, y_dev, writer=dev_summary_writer)
                            print("")

                # save on each epoch
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

if __name__ == '__main__':
    # Parameters
    # ==================================================

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")


    # Data Preparatopn
    # ==================================================

    # Load data
    print("Loading data...")
    x, y, vocabulary, vocabulary_inv = data_helpers.load_data()
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
    print("Vocabulary Size: {:d}".format(len(vocabulary)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


    # Training
    # ==================================================

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    dummy_embeddings = np.zeros((100,100))

    text_cnn_train_and_save_model(x_train=x_train, y_train=y_train,
                                  x_dev=x_dev, y_dev=y_dev,
                                  out_dir=out_dir,
                                  allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement,
                                  embeddings=dummy_embeddings,
                                  vocabulary=vocabulary,
                                  filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                                  num_filters=FLAGS.num_filters,
                                  l2_reg_lambda=FLAGS.l2_reg_lambda,
                                  dropout_keep_prob=FLAGS.dropout_keep_prob,
                                  batch_size=FLAGS.batch_size,
                                  num_epochs=FLAGS.num_epochs,
                                  evaluate_every=FLAGS.evaluate_every,
                                  checkpoint_every=FLAGS.checkpoint_every)
