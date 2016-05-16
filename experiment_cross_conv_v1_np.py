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


from multiprocessing import Pool


def single_func(s1s2):
    print 'Zipped item :'
    #print s1s2

    s1, s2 = zip(*s1s2)
    print s1
    print s2
    return convolve_cross_filter(s1, s2, filter_size)
    # print "batch %s of %s"%(i, batch_size)

def single_func1(s1s2):
    #print 'Zipped item :'
    #print s1s2
    s1u = s1s2['s1']
    s2u = s1s2['s2']
    fs = s1s2['fs']

    return convolve_cross_filter(s1u, s2u, fs)

def convolve_cross_filter_batch_multicore(s1, s2, filter_size, processes_cnt):
    batch_size, sent_len, embedding_size = s1.shape
    conv_iter = sent_len - filter_size + 1

    # batches_conv_res = np.zeros((batch_size, conv_iter, conv_iter))
    # shared_array = shared_array.reshape(batch_size, conv_iter, conv_iter)
    # shared_array = shared_array.reshape(10, 10)

    # s1s2_zipped = list(zip(s1, s2))
    s1s2_zipped = []
    for i in range(0, batch_size):
        s1s2_zipped.append({'s1': s1[i], 's2': s2[i], 'fs':filter_size})
    print 'Zipped:'
    #print s1s2_zipped
    pool = Pool(processes=processes_cnt)
    shared_array = pool.map(single_func1, s1s2_zipped)
    #pool.join()

    return np.array(shared_array)

import sys
if __name__ == '__main__':
    import logging  # word2vec logging

    # Set logging info
    logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s]: %(levelname)s : %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Enable file logging
    # logFileName = '%s/%s-%s.log' % ('logs', 'sup_parser_v1', '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now()))
    # fileHandler = logging.FileHandler(logFileName, 'wb')
    # fileHandler.setFormatter(logFormatter)
    # logger.addHandler(fileHandler)

    # Enable console logging
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    import time as ti
    batch_size = 20
    sent_len = 100
    embedding_size = 300

    threads = 20

    s1 = np.random.rand(batch_size, sent_len, embedding_size)
    s2 = np.random.rand(batch_size, sent_len, embedding_size)

    filter_size = 3

    # multiprocessing
    pool_size = 20

    logging.info('Parallel claculation with %s pools...' % pool_size)
    start = ti.time()


    batches_conv_res = convolve_cross_filter_batch_multicore(s1, s2, 3, pool_size)

    end = ti.time()
    logging.info('calculated in %s '%(end-start))

    print(batches_conv_res.shape)
    print(batches_conv_res)




