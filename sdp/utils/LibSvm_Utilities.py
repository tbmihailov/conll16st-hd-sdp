import sys
import pandas as pd
import operator
import re

class LibSvm_Utilities(object):
    @staticmethod
    def write_libsvm_file_from_vectors_list(list_of_vectors, classes_list, output_file_libsvm):
        with open(output_file_libsvm, "w") as libsvm_file:
            for row_idx in range(0, len(list_of_vectors)):
                row_class=classes_list[row_idx]
                libsvm_file.write('{:d}'.format(row_class))
                for f_idx in range(0,len(list_of_vectors[row_idx])):
                    libsvm_file.write('\t{:d}:{:f}'.format(f_idx+1, list_of_vectors[row_idx][f_idx]))
                libsvm_file.write('\n')


    @staticmethod
    def write_libsvm_file_from_sparsefeatures_list(list_of_sparse, classes_list, feat_dict, output_file_libsvm):
        feat_dict_sorted_tupl = sorted(feat_dict.iteritems(), key=lambda (k,v): (v,k))
        #feat_dict_sorted = {}
        #for k,v in feat_dict_sorted_tupl:
        #    feat_dict_sorted[k]=v

        print feat_dict_sorted_tupl
        with open(output_file_libsvm, "w") as libsvm_file:
            for row_idx in range(0, len(list_of_sparse)):
                row_class=classes_list[row_idx]
                libsvm_file.write('{:d}'.format(row_class))

                for key, val in feat_dict_sorted_tupl:
                    if key in list_of_sparse[row_idx]:
                        #print "Row:%s;Key:%s"%(row_idx, key)
                        #print list_of_sparse[row_idx][key]
                        libsvm_file.write('\t%d:%f'%(int(val), list_of_sparse[row_idx][key]))

                libsvm_file.write('\n')

    @staticmethod
    def write_libsvm_file_from_intkey_floatval_sparsefeatures_tuple_list(list_of_sparse, classes_list, output_file_libsvm, filter_includekeys=None, sort_features=False):
        with open(output_file_libsvm, "w") as libsvm_file:
            #print filter_includekeys
            for row_idx in range(0, len(list_of_sparse)):
                row_class=classes_list[row_idx]
                libsvm_file.write('{:d}'.format(row_class))
                #print list_of_sparse[row_idx]

                if sort_features:
                    feat_sorted_tupl = sorted(list_of_sparse[row_idx], key=lambda tup: tup[0])
                else:
                    feat_sorted_tupl = list_of_sparse[row_idx]

                if not filter_includekeys is None:
                    feat_sorted_tupl = [(x,y) for (x,y) in feat_sorted_tupl if x in filter_includekeys]

                #print feat_sorted_tupl
                for key, val in feat_sorted_tupl:
                    libsvm_file.write('\t%d:%f'%(key, val))

                libsvm_file.write('\n')

    @staticmethod
    def read_libsvm_features_into_intkey_floatval_sparsefeatures_dict_list(input_file_libsvm):
        data_dict_list=[]
        classes_list=[]

        with open(input_file_libsvm,'r') as libsvm_file:
            for line in libsvm_file:
                items =  re.split('\s', line.strip())
                classes_list.append(int(items[0]))
                values_dict = {}
                for i in range(1,len(items)):
                    item_kv = items[i].split(':')
                    values_dict[int(item_kv[0])] = float(item_kv[1])

                data_dict_list.append(values_dict)

        return classes_list, data_dict_list

    @staticmethod
    def read_libsvm_features_into_intkey_floatval_sparsefeatures_tuple_list(input_file_libsvm):
        data_dict_list=[]
        classes_list=[]

        with open(input_file_libsvm,'r') as libsvm_file:
            for line in libsvm_file:
                items =  re.split('\s', line.strip())
                classes_list.append(int(items[0]))
                values_dict = []
                for i in range(1,len(items)):
                    item_kv = items[i].split(':')
                    values_dict.append((int(item_kv[0]),float(item_kv[1])))

                data_dict_list.append(values_dict)

        return classes_list, data_dict_list



    @staticmethod
    def read_libsvm_predictions_file(libsvm_predictions_file, autoremove_headers=True):
        res_data = []
        with open(libsvm_predictions_file, "r") as f:
            for line in f:
                if(line.startswith("labels")):
                    continue
                res_data.append(line.replace("\n","").split(" "))

        return res_data

if __name__ == '__main__':
    sample_data = [[0.4, 0.5, 0.23, -0.14], \
    [0.33, 0.67, -0.36, -0.44], \
    [0.56, 0.5, -0.86, -0.54]]

    sample_data1 = [[0.2, 0.1, 0.123, -0.45], \
    [0.133, 0.47, -0.23, -0.11], \
    [0.16, 0.9, -0.36, -0.44]]

    classes = [0,1,0]
    output_file="text.libsvm"

    #combine arrays into feature vectors
    sample_data_export=[]
    for i in range(0, len(sample_data)):
        data_row = sample_data[i]+sample_data1[i]
        sample_data_export.append(data_row)

    #calculate similarity
    from scipy import spatial
    similarities=[]
    for i in range(0, len(sample_data)):
        similarity =1 - spatial.distance.cosine(sample_data[i],sample_data1[i])
        similarities.append(similarity)

    print "Similarities:"
    print similarities
    #export to livsvm format
    import time
    start = time.time()
    LibSvm_Utilities.write_libsvm_file_from_vectors_list(sample_data_export, classes, output_file)
    end = time.time()
    print("Done! Execution time: %s s"%(end - start))
