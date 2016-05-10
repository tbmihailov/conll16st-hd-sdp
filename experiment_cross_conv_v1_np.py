import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d

import multiprocessing
import ctypes
import numpy as np

def convolve_cross(s1, s2, filter_size):
    batch_size = s1.shape[0]
    sent_len = s1.shape[1]
    embedding_size = s1.shape[2]
    conv_iter = sent_len - filter_size + 1

    batches_conv_res = np.zeros((batch_size, conv_iter, conv_iter))
    for bi in range(0, batch_size):
        batch_s1 = s1[bi] #  tf.gather(s1, bi)
        batch_s2 = s2[bi] #  tf.gather(s1, bi)

        for i in range(0, conv_iter):
            filter_s1 = batch_s1[i:i+filter_size]
            for j in range(0, conv_iter):
                filter_s2 = batch_s2[j:j+filter_size]
                curr_val = 0
                for k in range(0, filter_size):
                    for l in range(0, embedding_size):
                        curr_val += filter_s1[k][l] * filter_s2[k][l]
                batches_conv_res[bi][i][j]

        print "batch %s of %s"%(bi, batch_size)
    return np.array(batches_conv_res)


def convolve_cross_filter(batch_s1, batch_s2, filter_size):
    sent_len, embedding_size = batch_s1.shape
    conv_iter = sent_len - filter_size + 1

    batch_res = np.zeros((conv_iter, conv_iter))
    for i in range(0, conv_iter):
        filter_s1 = batch_s1[i:i + filter_size]
        for j in range(0, conv_iter):
            filter_s2 = batch_s2[j:j + filter_size]
            curr_val = 0
            for k in range(0, filter_size):
                for l in range(0, embedding_size):
                    curr_val += filter_s1[k][l] * filter_s2[k][l]
            batch_res[i][j] = curr_val
    return batch_res

def convolve_cross_filter_batch(s1, s2, filter_size):
    batch_size, sent_len, embedding_size = s1.shape
    conv_iter = sent_len - filter_size + 1

    batches_conv_res = np.zeros((batch_size, conv_iter, conv_iter))
    for bi in range(0, batch_size):
        batches_conv_res[bi] = convolve_cross_filter(s1[bi], s2[bi], filter_size)

        print "batch %s of %s"%(bi, batch_size)
    return batches_conv_res

def convolve_cross_filter_batch_multicore(s1, s2, filter_size, processes_cnt):
    batch_size, sent_len, embedding_size = s1.shape
    conv_iter = sent_len - filter_size + 1

    shared_array_base = multiprocessing.Array(ctypes.c_double, batch_size*conv_iter*conv_iter)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())


    # batches_conv_res = np.zeros((batch_size, conv_iter, conv_iter))
    shared_array = shared_array.reshape(batch_size, conv_iter, conv_iter)
    # shared_array = shared_array.reshape(10, 10)

    def single_func(i):
        shared_array[i] = convolve_cross_filter(s1[i], s2[i], filter_size)
        # print "batch %s of %s"%(i, batch_size)

    pool = multiprocessing.Pool(processes=processes_cnt)
    pool.map(single_func, range(batch_size))

    return shared_array

if __name__ == '__main__':
    batch_size = 10
    sent_len = 15
    embedding_size = 20

    s1 = np.empty([batch_size, sent_len, embedding_size])
    s2 = np.empty([batch_size, sent_len, embedding_size])

    s1.fill(9)
    s2.fill(4)

    filter_size = 3

    batches_conv_res = np.array(convolve_cross_filter_batch_multicore(s1, s2, 3, 5))
    print(batches_conv_res.shape)
    print(batches_conv_res)




