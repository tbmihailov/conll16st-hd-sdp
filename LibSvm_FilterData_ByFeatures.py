import sys
import pandas as pd
import operator
import re
import sys
from LibSvm_Utilities import LibSvm_Utilities
from Common_Utilities import CommonUtilities

#SAMPLE USAGE
#set files_prefix=E:\semeval2016-task3-caq\data\semeval2016-task3-cqa-ql-traindev-v3.2\v3.2\train\SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml_tr1dev_veconly
#set features_filter_file=LibSvm_FilterData_ByFeatures.features_filter
#python LibSvm_FilterData_ByFeatures.py %files_prefix%.libsvm %files_prefix%.features %features_filter_file% %features_filter_file%.libsvm
if __name__ == '__main__':

    if len(sys.argv) > 0:
        input_libsvm_file = sys.argv[1]
        print('input_libsvm_file:\t%s' % input_libsvm_file)
    else:
        print('Error: missing input_libsvm_file parameter')
        quit()

    if len(sys.argv) > 0:
        features_dict_file = sys.argv[2]
        print('features_dict_file:\t%s' % features_dict_file)
    else:
        print('Error: missing features_dict_file parameter')
        quit()

    if len(sys.argv) > 0:
        features_list_filter_file = sys.argv[3]
        print('features_filter_file:\t%s' % features_list_filter_file)
    else:
        print('Error: missing features_filter_file parameter')
        quit()

    if len(sys.argv) > 0:
        output_libsvm_file = sys.argv[4]
        print('output_libsvm_file:\t%s' % output_libsvm_file)
    else:
        print('Error: missing output_libsvm_file parameter')
        quit()

    import time
    start = time.time()
    print "Reading data..."
    labels, sparse_data_intkey_floatval = LibSvm_Utilities.read_libsvm_features_into_intkey_floatval_sparsefeatures_tuple_list(input_libsvm_file)
    print "Done in %s s"%(time.time()-start)

    print "read data: %s items"%len(sparse_data_intkey_floatval)
    features_dict = CommonUtilities.load_dictionary_from_file(features_dict_file)
    features_dict = dict([(k,int(v)) for (k,v) in features_dict.items()])
    print "features_dict: %s items"%len(features_dict)

    features_filter_list=[]
    with open(features_list_filter_file,'r') as f:
        for line in f:
            features_filter_list.append(line.strip())

    features_indexes_to_include = []
    features_to_include=[]
    for k,v in features_dict.iteritems():
        for filt in features_filter_list:
            if k.startswith(filt):
                features_indexes_to_include.append(v)
                features_to_include.append(k)
    print "features_indexes_to_include: %s items"%len(features_indexes_to_include)
    print "features to include:"
    features_to_include = sorted(features_to_include, key=lambda x:x)
    for i in range(0, len(features_to_include)):
        print features_to_include[i]
    print "writing data..."
    start1 = time.time()
    LibSvm_Utilities.write_libsvm_file_from_intkey_floatval_sparsefeatures_tuple_list(sparse_data_intkey_floatval, labels, output_libsvm_file, features_indexes_to_include, False)
    print "Done in %s s"%(time.time()-start1)

    print "New file\n %s"%output_libsvm_file
    end = time.time()
    print("Done! Execution time: %s s"%(end - start))



