import tensorflow as tf
import numpy as np

batch_size = 50
sent_len = 100
embedding_size = 300

shape = [batch_size, sent_len, embedding_size]
s1_np = np.random.rand(batch_size, sent_len, embedding_size)
s2_np = np.random(batch_size, sent_len, embedding_size)

filter_size = 3
conv_iter = sent_len - filter_size +1


sess = tf.InteractiveSession()

s1 = tf.placeholder(tf.int32, [None, embedding_size], name="s1")
s2_filters = tf.placeholder(tf.int32, [None, embedding_size], name="s2_filters")

# s2 = tf.fill([batch_size, sent_len, embedding_size], 4)

filter_size = 3
conv_iter = sent_len - filter_size +1

c = tf.zeros([batch_size, conv_iter, conv_iter])

from tensorflow_custom_op import tensor_to_filter_3


elems = tf.constant([1,2,3,4,5,6,7,8])

res = sess.run(tensor_to_filter_3(elems))

print res



