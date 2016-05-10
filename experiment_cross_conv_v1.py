import tensorflow as tf


sess = tf.InteractiveSession()

batch_size = 10
sent_len = 15
embedding_size = 20

s1 = tf.fill([batch_size, sent_len, embedding_size], 9)
s2 = tf.fill([batch_size, sent_len, embedding_size], 4)

filter_size = 3
conv_iter = sent_len - filter_size +1

c = tf.zeros([batch_size, conv_iter, conv_iter])

from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops


def tf_convolve(s1, s2, filter_size, conv_iter, embedding_size, batch_size):

    # batch_size = tf.shape(s1)[0] # Problem here?
    # conv_iter = tf.shape(s1)[1] - filter_\ize + 1
    # embedding_size = tf.shape(s1)[2]
    batches_conv_res = tf.Variable(tf.zeros([1, conv_iter, conv_iter]))

    for bi in range(0, batch_size):
        batch_s1 = tf.gather(s1, bi)
        batch_s2 = tf.gather(s2, bi)

        conv_res = tf.Variable(tf.zeros([conv_iter]))
        for i in range(0, conv_iter):
            filter_s1 = tf.slice(batch_s1, i, i+filter_size) # tf.slice
            conv_res_row = tf.Variable(tf.zeros([1]))
            for j in range(0, conv_iter):
                filter_s2 = tf.slice(batch_s2, j, j+filter_size)
                curr_val = tf.Variable(0)
                for k in range(0, filter_size):
                    for l in range(0, embedding_size):
                        curr_val += filter_s1[k][l] * filter_s2[k][l]

                tf.concat(conv_res_row, curr_val) #  conv_res_row.append(curr_val)

            tf.concat(conv_res, conv_res_row) #  conv_res.append(conv_res_row)

        tf.concat(batches_conv_res, conv_res)# batches_conv_res.append(conv_res)

    return batches_conv_res

#batches_conv = tf_convolve(s1, s2, filter_size, conv_iter, embedding_size, batch_size)
#batches_conv_val = sess.run(batches_conv)

# print(batches_conv_val)







