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

from tensorflow_custom_op import tensor_to_filter_3


elems = tf.constant([1,2,3,4,5,6,7,8])

res = sess.run(tensor_to_filter_3(elems))

print res



