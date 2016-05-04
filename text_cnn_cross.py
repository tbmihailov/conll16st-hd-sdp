import tensorflow as tf
import numpy as np


class TextCNNModel_Cross(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
        self, sequence_length,
            num_classes,
            vocab_size,
            embeddings,
            filter_sizes, num_filters,
            l2_reg_lambda=0.0,
            filter_sizes_cross=[3,4,5]):

        # Placeholders for input, output and dropout
        self.input_x_s1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_s1")
        self.input_x_s2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_s2")

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     W = tf.Variable(
        #         tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        #         name="W")
        #     self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        #     self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        self.embedding_size = embeddings.shape[1]
        self.embeddings_number = embeddings.shape[0]
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embeddings_placeholder = tf.placeholder(tf.float32, shape=[self.embeddings_number, self.embedding_size])
            # embeddings_tuned = tf.Variable(embeddings_placeholder)

            self.embedded_chars_s1 = tf.nn.embedding_lookup(self.embeddings_placeholder, self.input_x_s1)
            self.embedded_chars_expanded_s1 = tf.expand_dims(self.embedded_chars_s1, -1)

            self.embedded_chars_s2 = tf.nn.embedding_lookup(self.embeddings_placeholder, self.input_x_s2)
            self.embedded_chars_expanded_s2 = tf.expand_dims(self.embedded_chars_s2, -1)

            # self.embedded_chars_s1s2_matmul = tf.batch_matmul(self.embedded_chars_expanded_s1, self.embedded_chars_expanded_s2)
            # self.embedded_chars_s1s2_matmul_expanded = tf.expand_dims(self.embedded_chars_s1s2_matmul, -1)


        # Create a convolution + maxpool layer for each s2 filter size
        pooled_outputs_cross_s1s2 = []

        num_filters_cross = sequence_length - 1
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("cross-conv-maxpool-%s" % filter_size):
                # Convolution Layer

                filter_shape = [filter_size, self.embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # conv = tf.nn.conv2d(
                #     self.embedded_chars_s1,
                #     W,
                #     strides=[1, 1, 1, 1],
                #     padding="VALID",
                #     name="conv")

                #S1
                conv_s1 = tf.nn.conv2d(
                    self.embedded_chars_expanded_s1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_s1")

                # Apply nonlinearity
                h_s1 = tf.nn.relu(tf.nn.bias_add(conv_s1, b), name="relu_s1")

                # Maxpooling over the outputs
                pooled_cross_s1 = tf.nn.max_pool(
                    h_s1,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_1")
                pooled_outputs_cross_s1s2.append(pooled_cross_s1)

                #S2
                conv_s2 = tf.nn.conv2d(
                    self.embedded_chars_expanded_s2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_s2")

                # Apply nonlinearity
                h_s2 = tf.nn.relu(tf.nn.bias_add(conv_s2, b), name="relu_s2")

                # Maxpooling over the outputs
                pooled_cross_s2 = tf.nn.max_pool(
                    h_s2,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_s2")

                pooled_outputs_cross_s1s2.append(pooled_cross_s2)

        pooled_outputs = []
        # # Create a convolution + maxpool layer for each filter size
        # pooled_outputs = []
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.name_scope("conv-maxpool-%s" % filter_size):
        #         # Convolution Layer
        #         filter_shape = [filter_size, self.embedding_size, 1, num_filters]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        #         conv = tf.nn.conv2d(
        #             self.embedded_chars_expanded,
        #             W,
        #             strides=[1, 1, 1, 1],
        #             padding="VALID",
        #             name="conv")
        #         # Apply nonlinearity
        #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #         # Maxpooling over the outputs
        #         pooled = tf.nn.max_pool(
        #             h,
        #             ksize=[1, sequence_length - filter_size + 1, 1, 1],
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name="pool")
        #         pooled_outputs.append(pooled)

        pooled_outputs.extend(pooled_outputs_cross_s1s2)

        # Combine all the pooled features
        num_filters_total = 2 * num_filters * len(filter_sizes) # multiplied by 2
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class TextCNNModel_Cross_v1(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
        self, sequence_length,
            num_classes,
            vocab_size,
            embeddings,
            filter_sizes, num_filters,
            l2_reg_lambda=0.0,
            filter_sizes_cross=[3,4,5]):

        # Placeholders for input, output and dropout
        self.input_x_s1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_s1")
        self.input_x_s2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_s2")

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     W = tf.Variable(
        #         tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        #         name="W")
        #     self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        #     self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        self.embedding_size = embeddings.shape[1]
        self.embeddings_number = embeddings.shape[0]
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embeddings_placeholder = tf.placeholder(tf.float32, shape=[self.embeddings_number, self.embedding_size])
            # embeddings_tuned = tf.Variable(embeddings_placeholder)

            self.embedded_chars_s1 = tf.nn.embedding_lookup(self.embeddings_placeholder, self.input_x_s1)
            self.embedded_chars_expanded_s1 = tf.expand_dims(self.embedded_chars_s1, -1)

            self.embedded_chars_s2 = tf.nn.embedding_lookup(self.embeddings_placeholder, self.input_x_s2)
            self.embedded_chars_expanded_s2 = tf.expand_dims(self.embedded_chars_s2, -1)

            # self.embedded_chars_s1s2_matmul = tf.batch_matmul(self.embedded_chars_expanded_s1, self.embedded_chars_expanded_s2)
            # self.embedded_chars_s1s2_matmul_expanded = tf.expand_dims(self.embedded_chars_s1s2_matmul, -1)

        # Create a convolution + maxpool layer for each s2 filter size
        pooled_outputs_cross_s1s2 = []

        num_filters_cross = sequence_length - 1
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("cross-conv-maxpool-%s" % filter_size):
                # Convolution Layer

                filter_shape = [filter_size, self.embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # conv = tf.nn.conv2d(
                #     self.embedded_chars_s1,
                #     W,
                #     strides=[1, 1, 1, 1],
                #     padding="VALID",
                #     name="conv")

                #S1
                conv_s1 = tf.nn.conv2d(
                    self.embedded_chars_expanded_s1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_s1")

                # Apply nonlinearity
                h_s1 = tf.nn.relu(tf.nn.bias_add(conv_s1, b), name="relu_s1")

                # Maxpooling over the outputs
                pooled_cross_s1 = tf.nn.max_pool(
                    h_s1,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_1")
                pooled_outputs_cross_s1s2.append(pooled_cross_s1)

                #S2
                conv_s2 = tf.nn.conv2d(
                    self.embedded_chars_expanded_s2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_s2")

                # Apply nonlinearity
                h_s2 = tf.nn.relu(tf.nn.bias_add(conv_s2, b), name="relu_s2")

                # Maxpooling over the outputs
                pooled_cross_s2 = tf.nn.max_pool(
                    h_s2,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_s2")

                pooled_outputs_cross_s1s2.append(pooled_cross_s2)

        pooled_outputs = []
        # # Create a convolution + maxpool layer for each filter size
        # pooled_outputs = []
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.name_scope("conv-maxpool-%s" % filter_size):
        #         # Convolution Layer
        #         filter_shape = [filter_size, self.embedding_size, 1, num_filters]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        #         conv = tf.nn.conv2d(
        #             self.embedded_chars_expanded,
        #             W,
        #             strides=[1, 1, 1, 1],
        #             padding="VALID",
        #             name="conv")
        #         # Apply nonlinearity
        #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #         # Maxpooling over the outputs
        #         pooled = tf.nn.max_pool(
        #             h,
        #             ksize=[1, sequence_length - filter_size + 1, 1, 1],
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name="pool")
        #         pooled_outputs.append(pooled)

        pooled_outputs.extend(pooled_outputs_cross_s1s2)

        # Combine all the pooled features
        num_filters_total = 2 * num_filters * len(filter_sizes) # multiplied by 2
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
