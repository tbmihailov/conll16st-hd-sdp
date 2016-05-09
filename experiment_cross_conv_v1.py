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

def tf_convolve(s1,s2, filter_size):
    batch_size = tf.shape(s1)[0] # Problem here?
    conv_iter = tf.shape(s1)[1] - filter_size + 1
    embedding_size = tf.shape(s1)[2]
    batches_conv_res = tf.variable(tf.float32, [None, None])
    for bi in range(0, batch_size):
        batch_s1 = tf.gather(s1, bi)
        batch_s2 = tf.gather(s2, bi)

        conv_res = tf.variable(tf.float32, [None, None])
        for i in range(0, conv_iter):
            filter_s1 = tf.slice(batch_s1, i, i+filter_size) # tf.slice
            conv_res_row = tf.variable(tf.float32, [None, None])
            for j in range(0, conv_iter):
                filter_s2 = tf.slice(batch_s2, j, j+filter_size)
                curr_val = tf.variable(tf.float32)
                for k in range(0, filter_size):
                    for l in range(0, embedding_size):
                        curr_val += filter_s1[k][l] * filter_s2[k][l]
                conv_res_row.append(curr_val)

            conv_res.append(conv_res_row)

        batches_conv_res.append(conv_res)

    batches_conv_res

batches_conv = tf_convolve(s1, s2)
batches_conv_res = sess.run(batches_conv)

print(batches_conv_res)







