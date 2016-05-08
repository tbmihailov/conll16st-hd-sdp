import numpy as np



batch_size = 10
sent_len = 15
embedding_size = 20

s1 = np.empty([batch_size, sent_len, embedding_size])
s2 = np.empty([batch_size, sent_len, embedding_size])

s1.fill(9)
s2.fill(4)

filter_size = 3
conv_iter = sent_len - filter_size +1

c = np.zeros([batch_size, conv_iter, conv_iter])

def convolve(s1,s2):
    batches_conv_res = []
    for bi in range(0, batch_size):
        batch_s1 = s1[bi] #  tf.gather(s1, bi)
        batch_s2 = s2[bi] #  tf.gather(s1, bi)

        conv_res = []
        for i in range(0, conv_iter):
            filter_s1 = batch_s1[i:i+filter_size]
            conv_res_row = []
            for j in range(0, conv_iter):
                filter_s2 = batch_s2[j:j+filter_size]
                curr_val = 0
                for k in range(0, filter_size):
                    for l in range(0, embedding_size):
                        curr_val += filter_s1[k][l] * filter_s2[k][l]
                conv_res_row.append(curr_val)

            conv_res.append(conv_res_row)

        batches_conv_res.append(conv_res)

    batches_conv_res

batches_conv_res = convolve(s1, s2)
print(batches_conv_res)




