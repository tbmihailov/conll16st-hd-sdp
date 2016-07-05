import sys
import pandas as pd
import operator
import re
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from LibSvm_Utilities import LibSvm_Utilities
from Common_Utilities import CommonUtilities

#SAMPLE USAGE
# set prefix=svm_base_sup_v2_hier_ext_tr16dev
# set input_libsvm_file=${prefix}_scalerange__EXP_LEVEL1_scale_train.libsvm
# set features_filter_file=features_filter_nonexp_all.features_filter
# set features_dict_file=${prefix}_model__NONEXP_LEVEL1.features
# set class_mappings_file=${prefix}_model_.classmapping
# python LibSvm_Data_FeatureSelection.py -input_libsvm_file:${input_libsvm_file} -features_dict_file:${features_dict_file} -features_filter_file:${features_filter_file} -class_mappings_file:${class_mappings_file}
if __name__ == '__main__':

    input_libsvm_file=''
    input_libsvm_file = CommonUtilities.get_param_value('input_libsvm_file', sys.argv, default=input_libsvm_file)
    # if len(input_libsvm_file) == 0:
    #     print('Error: missing input_libsvm_file parameter')
    #     quit()

    features_dict_file = ''
    features_dict_file = CommonUtilities.get_param_value('features_dict_file', sys.argv, default=features_dict_file)
    # if len(features_dict_file) == 0:
    #     print('Error: missing features_dict_file parameter')
    #     quit()

    features_filter_file= ''
    features_filter_file = CommonUtilities.get_param_value('features_filter_file', sys.argv, default=features_filter_file)
    # if len(features_filter_file) == 0:
    #     print('Error: missing features_filter_file parameter')
    #     quit()

    class_mappings_file=''
    class_mappings_file = CommonUtilities.get_param_value('class_mappings_file', sys.argv, default=class_mappings_file)
    # if len(class_mappings_file) == 0:
    #     print('Error: missing class_mappings_file parameter')
    #     quit()

    import time
    start = time.time()
    print "Reading data..."
    #labels, sparse_data_intkey_floatval = LibSvm_Utilities.read_libsvm_features_into_intkey_floatval_sparsefeatures_tuple_list(input_libsvm_file)
    labels, sparse_data_intkey_floatval = LibSvm_Utilities.read_libsvm_features_into_intkey_floatval_sparsefeatures_dict_list(
        input_libsvm_file)
    print "Done in %s s"%(time.time()-start)
    print "read data: %s items"%len(sparse_data_intkey_floatval)

    features_dict = CommonUtilities.load_dictionary_from_file(features_dict_file)
    features_dict = dict([(k,int(v)) for (k,v) in features_dict.items()])
    print "features_dict: %s items"%len(features_dict)

    class_mappings_dict = CommonUtilities.load_dictionary_from_file(class_mappings_file)
    class_mappings_dict = dict([(k,int(v)) for (k,v) in class_mappings_dict.items()])
    print "class_mappings_dict: %s items"%len(class_mappings_dict)

    #read filters
    if features_filter_file!='':
        print('Filtering file:%s'%features_filter_file)
        features_filter_list=[]
        with open(features_filter_file, 'r') as f:
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

        start1 = time.time()
        print("Filtering features from %s items"%len(sparse_data_intkey_floatval))
        for item in sparse_data_intkey_floatval:
            for k in item.iterkeys():
                if not k in features_indexes_to_include:
                    del item[k]

        print("Done! Execution time: %s s" % (time.time() - start1))

    #vectorizer
    from sklearn.feature_extraction import DictVectorizer
    v = DictVectorizer(sparse=False)

    data = v.fit_transform(sparse_data_intkey_floatval)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    print "Scaling..."
    start = time.time()
    data = scaler.fit_transform(data)

    # # feature selection
    #========================================================
    # print "Feature selection using ExtraTreesClassifier..."
    # start1 = time.time()
    # from sklearn.ensemble import ExtraTreesClassifier
    # model = ExtraTreesClassifier()
    # model.fit(data, labels)
    # print("Done! Execution time: %s s" % (time.time() - start1))
    # print "Feature importance:"
    # print model.feature_importances_

    #===================================
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    #from sklearn.datasets import make_classification

    # Build a classification task using 3 informative features
    # X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
    #                            n_redundant=2, n_repeated=0, n_classes=8,
    #                            n_clusters_per_class=1, random_state=0)

    # Create the RFE object and compute a cross-validated score.
    classifier_current = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, fit_intercept=True,
                                            intercept_scaling=1, random_state=None,
                                            solver='liblinear',
                                            max_iter=100, multi_class='ovr', verbose=0, warm_start=False,
                                            n_jobs=8)
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=classifier_current, step=1, cv=StratifiedKFold(labels, 5),
                  scoring='accuracy')
    rfecv.fit(data, labels)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()



