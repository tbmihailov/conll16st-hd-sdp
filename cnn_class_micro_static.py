import sys
import numpy as np
import tensorflow as tf
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import metrics


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


class TextCNN(object):
    def __init__(self, train_dataset, train_labels, valid_dataset, valid_labels, embeddings, vocabulary, l2_reg_lambda,
                 num_steps, batch_size, num_filters, filter_sizes_1, filter_sizes_2, filter_sizes_3, dropout_keep_prob,
                 #  lexical,
                 shuffling, num_classes):
        # parameters
        vocab_size = len(vocabulary)
        # sequence_length = train_dataset.shape[1]
        sequence_length = train_dataset.shape[1]

        train_size = train_dataset.shape[0]
        # num_classes = 3

        filter_sizes = [filter_sizes_1, filter_sizes_2, filter_sizes_3]
        num_filters_total = num_filters * len(filter_sizes)

        embedding_size = embeddings.shape[1]
        # more embeddings than words in vocab :/
        embeddings_number = embeddings.shape[0]

        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(10)
            # variables and constants
            input_x = tf.placeholder(tf.int32, shape=[batch_size, sequence_length], name="input_x")
            input_y = tf.placeholder(tf.int32, shape=[batch_size, num_classes], name="input_y")

            # tf_valid_dataset = tf.constant(valid_dataset)
            tf_valid_dataset = tf.placeholder(tf.int32, shape=[None, sequence_length])

            reg_coef = tf.placeholder(tf.float32)
            l2_loss = tf.constant(0.0)

            # Generate convolution weights. This should be more human readable
            weights_conv = [tf.Variable(tf.truncated_normal([filter_size, embedding_size, 1, num_filters],
                                                            stddev=tf.sqrt(2.0 / (filter_size * embedding_size)),
                                                            seed=filter_size + i * num_filters)) for i, filter_size in
                            enumerate(filter_sizes)]
            # weights_conv = [tf.Variable(tf.truncated_normal([filter_size, embedding_size, 1, num_filters], stddev = 0.1 , seed = filter_size + i*num_filters)) for i, filter_size in enumerate(filter_sizes)]
            biases_conv = [tf.Variable(tf.constant(0.01, shape=[num_filters])) for filter_size in filter_sizes]
            # biases_conv = [tf.Variable(tf.constant(0.1, shape=[num_filters])) for filter_size in filter_sizes]

            weight_output = tf.Variable(tf.truncated_normal([num_filters_total, num_classes],
                                                            stddev=tf.sqrt(2.0 / (num_filters_total + num_classes)),
                                                            seed=0))
            # weight_output = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev = 0.1, seed = 0))
            bias_output = tf.Variable(tf.constant(0.01, shape=[num_classes]))
            # bias_output = tf.Variable(tf.constant(0.1, shape=[num_classes]))

            embeddings_const = tf.placeholder(tf.float32, shape=[embeddings_number, embedding_size])
            # embeddings_tuned = tf.Variable(embeddings_placeholder)

            embedded_chars = tf.nn.embedding_lookup(embeddings_const, input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

            embedded_chars_valid = tf.nn.embedding_lookup(embeddings_const, tf_valid_dataset)
            embedded_chars_expanded_valid = tf.expand_dims(embedded_chars_valid, -1)

            def model(data, dropout_prob):
                pooled_outputs = []
                # lookup table
                for i, filter_size in enumerate(filter_sizes):
                    # convolution layer with different filter size
                    conv = tf.nn.conv2d(data, weights_conv[i], strides=[1, 1, 1, 1], padding="VALID")
                    # non-linearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, biases_conv[i]))
                    pooled = tf.nn.max_pool(h,
                                            ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID')
                    pooled_outputs.append(pooled)

                h_pool = tf.concat(3, pooled_outputs)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
                h_drop = tf.nn.dropout(h_pool_flat, dropout_prob)
                scores = tf.nn.xw_plus_b(h_drop, weight_output, bias_output)

                return scores

            scores = model(embedded_chars_expanded, dropout_keep_prob)
            train_prediction = tf.nn.softmax(scores)

            losses = tf.nn.softmax_cross_entropy_with_logits(scores, tf.cast(input_y, tf.float32))

            for i in range(len(weights_conv)):
                l2_loss += tf.nn.l2_loss(weights_conv[i])
            l2_loss += tf.nn.l2_loss(weight_output)

            loss = tf.reduce_mean(losses) + reg_coef * l2_loss

            # global_step = tf.Variable(0)
            # learning_rate = tf.train.exponential_decay(1e-4, global_step * batch_size, tf.size(input_x), 0.95, staircase=True)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            global_step = tf.Variable(0, trainable=False)
            # optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

            optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

            valid_prediction = tf.nn.softmax(model(embedded_chars_expanded_valid, 1.0))

        with tf.Session(graph=graph) as session:
            session.run(tf.initialize_all_variables())
            print ("Initialized")

            if (shuffling == "y"):
                np.random.seed(42)
                train = np.asarray(list(zip(train_dataset, train_labels)))
                np.random.shuffle(train)
                train_dataset, train_labels = zip(*train)
                train_dataset = np.asarray(train_dataset)
                train_labels = np.asarray(train_labels)

            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size)]
                batch_labels = train_labels[offset:(offset + batch_size)]
                feed_dict = {input_x: batch_data, input_y: batch_labels, reg_coef: l2_reg_lambda,
                             embeddings_const: embeddings}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict)
                if not step % 100:
                    print ("Minibatch loss at step", step, ":", l)
                    print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("\n")

            self.valid_predictions = session.run([valid_prediction], feed_dict={embeddings_const: embeddings, tf_valid_dataset: valid_dataset})
            self.valid_predictions = np.asarray(self.valid_predictions).reshape(valid_labels.shape)
            self.valid_accuracy = accuracy(self.valid_predictions, np.asarray(valid_labels))
