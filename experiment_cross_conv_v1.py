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

for bi in range(0, batch_size):
    batch_s1 = tf.gather(s1, bi)
    batch_s2 = tf.gather(s1, bi)

    filters_s1 = []

    #
    # batch_res = []
    # for i in range(0, conv_iter):
    #     line = []
    #     for j in range(0, conv_iter):
    #
    #         line.append(res_val)
    #
    #     batch_res.append()



