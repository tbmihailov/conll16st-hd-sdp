#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sample Discourse Relation Classifier Train

Train parser for suplementary evaluation

Train should take three arguments

	$inputDataset = the folder of the dataset to parse.
		The folder structure is the same as in the tar file
		$inputDataset/parses.json
		$inputDataset/relations-no-senses.json

	$inputRun = the folder that contains the word2vec_model file or other resources

	$outputDir = the folder that the parser will output 'output.json' to

"""

import codecs
import json
import random
import sys
from datetime import datetime

import logging  # word2vec logging

from sklearn import preprocessing

import validator
from Common_Utilities import CommonUtilities

import gensim
from gensim import corpora, models, similarities  # used for word2vec
from gensim.models.word2vec import Word2Vec  # used for word2vec
from gensim.models.doc2vec import Doc2Vec  # used for doc2vec

import time  # used for performance measuring
import math

from scipy import spatial  # used for similarity calculation
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Phrases

from gensim import corpora  # for dictionary
from gensim.models import LdaModel

# from sklearn.svm import libsvm
from sklearn.svm import SVC

from VocabEmbedding_Utilities import VocabEmbeddingUtilities
from infer import Embeddings
from sklearn.linear_model import LogisticRegression

sys.path.append('~/semanticz')
from Word2Vec_AverageVectorsUtilities import AverageVectorsUtilities

import pickle

from DiscourseSenseClassification_FeatureExtraction_v1 import DiscourseSenseClassification_FeatureExtraction


class DiscourseSenseClassifier_Sup_v2_Optimized_Hierarchical(object):
    """Sample discourse relation sense classifier
    """

    def __init__(self, valid_senses, input_run, input_dataset, output_dir, input_params, input_features, class_mapping
                 , scale_range=(-1, 1)):
        self.valid_senses = valid_senses
        self.input_run = input_run
        self.input_dataset = input_dataset
        self.output_dir = output_dir
        self.input_params = input_params
        self.input_features = input_features
        self.class_mapping = class_mapping
        self.scale_range = scale_range

        pass

    def train_sense(self, input_dataset, word2vec_model, save_model_file_basename, scale_features, deps_model,
                    save_scale_file_basename, hierachical_classifier=False):
        class_mapping_flat = self.class_mapping

        # Classes:
        # 'Temporal.Asynchronous.Precedence',
        # 'Temporal.Asynchronous.Succession',
        # 'Temporal.Synchrony',
        # 'Contingency.Cause.Reason',
        # 'Contingency.Cause.Result',
        # 'Contingency.Condition',
        # 'Comparison.Contrast',
        # 'Comparison.Concession',
        # 'Expansion.Conjunction',
        # 'Expansion.Instantiation',
        # 'Expansion.Restatement',
        # 'Expansion.Alternative',
        # 'Expansion.Alternative.Chosen alternative',
        # 'Expansion.Exception',
        # 'EntRel',

        class_tree = {'Expansion':
                          {'ID': 1,
                           'SubClasses':
                               {
                                   'Expansion.Conjunction': {'ID': 11},
                                   'Expansion.Instantiation': {'ID': 12},
                                   'Expansion.Restatement': {'ID': 13},
                                   'Expansion.Alternative': {'ID': 14},
                                   'Expansion.Alternative.Chosen alternative': {'ID': 15},
                                   'Expansion.Exception': {'ID': 16},
                               }
                           },
                      'Temporal':
                          {'ID': 2,
                           'SubClasses':
                               {
                                   'Temporal.Asynchronous.Precedence': {'ID': 21},
                                   'Temporal.Asynchronous.Succession': {'ID': 22},
                                   'Temporal.Synchrony': {'ID': 23},
                               }
                           },
                      'Contingency':
                          {'ID': 3,
                           'SubClasses':
                               {
                                   'Contingency.Cause.Reason': {'ID': 31},
                                   'Contingency.Cause.Result': {'ID': 32},
                                   'Contingency.Condition': {'ID': 33},
                               }
                           },
                      'Comparison':
                          {'ID': 4,
                           'SubClasses':
                               {
                                   'Comparison.Contrast': {'ID': 41},
                                   'Comparison.Concession': {'ID': 42},
                               }
                           },
                      'EntRel': {'ID': 5},
                      }

        logging.debug(class_mapping_flat)
        word2vec_index2word_set = set(word2vec_model.index2word)
        # model_dir = self.input_run

        relation_file = '%s/relations.json' % input_dataset  # with senses to train
        relation_dicts = [json.loads(x) for x in open(relation_file)]

        parse_file = '%s/parses.json' % input_dataset
        parse = json.load(codecs.open(parse_file, encoding='utf8'))

        # FEATURE EXTRACTION
        train_x = []
        train_y = []
        train_y_txt_level2 = []
        train_y_txt_level1 = []
        train_y_relation_types = []  # 1 Explicit, 0 Non-explicit

        logging.info('=====EXTRACTING FEATURES======')

        logging.info('Extracting features from %s items..' % len(train_x))
        for i, relation_dict in enumerate(relation_dicts):

            curr_features_vec = DiscourseSenseClassification_FeatureExtraction.extract_features_as_vector_from_single_record_v1( \
                relation_dict=relation_dict, \
                parse=parse, \
                word2vec_model=word2vec_model, \
                word2vec_index2word_set=word2vec_index2word_set, \
                deps_model=deps_model, \
                deps_vocabulary=set(deps_model._vocab),
            )

            if (i + 1) % 1000 == 0:
                print '%s of %s' % (i, len(relation_dicts))
                logging.info('%s of %s' % (i, len(relation_dicts)))
                print '%s features:%s' % (i, curr_features_vec[:10])

            curr_senses = relation_dict['Sense']  # list of senses example: u'Sense': [u'Contingency.Cause.Reason']
            # logging.debug('%s - %s'%(i, curr_senses))

            for curr_sense in curr_senses:
                train_x.append(curr_features_vec)
                train_y.append(0)

                train_y_txt_level2.append(curr_sense)
                if '.' in curr_sense:
                    train_y_txt_level1.append(curr_sense.split('.')[0])
                else:
                    train_y_txt_level1.append(curr_sense)

                if relation_dict['Type'] == 'Explicit':
                    train_y_relation_types.append(1)
                else:
                    train_y_relation_types.append(0)


        #SCALE FEATURES
        logging.info('=====SCALING======')
        scaler = preprocessing.MinMaxScaler(self.scale_range)
        if scale_features:
            logging.info('Scaling %s items with %s features..' % (len(train_x), len(train_x[0])))
            start = time.time()
            train_x = scaler.fit_transform(train_x)
            end = time.time()
            logging.info("Done in %s s" % (end - start))
            pickle.dump(scaler, open(save_scale_file_basename, 'wb'))
            logging.info('Scale feats ranges saved to %s' % save_scale_file_basename)
        else:
            logging.info("No scaling!")

        logging.info('======HIERARCHICAL TRAINING======')

        def filter_items_train_classifier_and_save_model(classifier_name, class_mapping_curr, relation_type, train_x, train_y_txt,
                                                         train_y_relation_types, save_model_file):
            """
            Filters items by given params, trains the classifier and saves the word2vec_model to a file.
            Args:
                classifier_name: Name of the classifier used for saving the models
                class_mapping_curr: Class mapping to map train_y_txt to int. Filters items
                relation_type: 1 Explicit, 0 Non Explicit, Filters items with this relation type only
                train_x: Train samples
                train_y_txt: Train sample classes - Text class that will be filtered using class_mapping_curr dict
                train_y_relation_types: Train type indicators if sample is explicit or implicit.
                Only items with relation_type will be used for training
                save_model_file: Name of the file in which the word2vec_model will be saved
            Returns:
                Filters items and trains classifier
            """
            logging.info('======[%s] - filter_items_train_classifier_and_save_model======' % classifier_name)

            train_x_curr = []
            train_y_curr = []

            # Filtering items
            logging.info('Filtering %s items...' % len(train_x))
            start = time.time()
            for i in range(0, len(train_x)):
                if train_y_txt[i] in class_mapping_curr and train_y_relation_types[i] == relation_type:
                    train_x_curr.append(train_x[i])
                    train_y_curr.append(class_mapping_curr[train_y_txt[i]])
            end = time.time()
            logging.info("Done in %s s" % (end - start))

            # Training
            # Classifier params
            # classifier_current = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            #                          degree=3, gamma='auto', kernel='rbf',
            #                          max_iter=-1, probability=False, random_state=None, shrinking=True,
            #                          tol=0.001, verbose=False)

            param_c=0.1
            classifier_current = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=param_c, fit_intercept=True,
                               intercept_scaling=1, class_weight=None, random_state=None,
                               solver='liblinear',
                               max_iter=100, multi_class='ovr', verbose=0, warm_start=False,
                               n_jobs=8)
            print 'Classifier:\n%s' % classifier_current

            start = time.time()
            logging.info('Training with %s items...' % len(train_x_curr))
            classifier_current.fit(train_x_curr, train_y_curr)
            end = time.time()
            logging.info("Done in %s s" % (end - start))

            # Saving word2vec_model
            pickle.dump(classifier_current, open(save_model_file, 'wb'))
            logging.info('Model saved to %s' % save_model_file)

        ###########################
        ### FILTER AND TRAIN ######
        ###########################

        # Classifier: Explicit, Level 1
        relation_type = 1  # 1 Explicit, 0 Non-Explicit, -1 All
        classifier_name = 'EXP_LEVEL1'
        # class_mapping_curr = dict([(k, v['ID']) for k, v in class_tree.iteritems()])
        class_mapping_curr = class_mapping_flat
        save_model_file_classifier_current = '%s_%s.modelfile' % (save_model_file_basename, classifier_name)

        filter_items_train_classifier_and_save_model(classifier_name=classifier_name,
                                        class_mapping_curr=class_mapping_curr,
                                        relation_type=relation_type,
                                        train_x = train_x,
                                        train_y_txt=train_y_txt_level2,
                                                     train_y_relation_types=train_y_relation_types,
                                        save_model_file=save_model_file_classifier_current)


        # Classifier: Non-Explicit, Level 1
        relation_type = 0  # 1 Explicit, 0 Non-Explicit, -1 All
        classifier_name = 'NONEXP_LEVEL1'
        # class_mapping_curr = dict([(k, v['ID']) for k, v in class_tree.iteritems()])
        class_mapping_curr = class_mapping_flat
        save_model_file_classifier_current = '%s_%s.modelfile' % (save_model_file_basename, classifier_name)

        filter_items_train_classifier_and_save_model(classifier_name=classifier_name,
                                                     class_mapping_curr=class_mapping_curr,
                                                     relation_type=relation_type,
                                                     train_x=train_x,
                                                     train_y_txt=train_y_txt_level2,
                                                     train_y_relation_types=train_y_relation_types,
                                                     save_model_file=save_model_file_classifier_current)

    def classify_sense(self, input_dataset, word2vec_model, load_model_file_basename, scale_features, deps_model,
                       load_scale_file_basename, hierachical_classifier=False):
        output_dir = self.output_dir

        class_mapping = self.class_mapping
        class_mapping_id_to_origtext = dict([(value, key) for key, value in class_mapping.iteritems()])
        logging.debug('class_mapping_id_to_origtext:')
        logging.debug(class_mapping_id_to_origtext)

        word2vec_index2word_set = set(word2vec_model.index2word)

        relation_file = '%s/relations-no-senses.json' % input_dataset
        parse_file = '%s/parses.json' % input_dataset
        parse = json.load(codecs.open(parse_file, encoding='utf8'))

        relation_dicts = [json.loads(x) for x in open(relation_file)]

        output_file = '%s/output.json' % output_dir
        output = codecs.open(output_file, 'wb', encoding='utf8')


        if scale_features:
            # scaler = preprocessing.MinMaxScaler(self.scale_range)
            # scaler.transform(feats)
            scaler = pickle.load(open(load_scale_file_basename, 'rb'))
            logger.info('Scaling is enabled!')
        else:
            logger.info('NO scaling!')


        # Classifier: Explicit, Level 1
        relation_type = 1  # 1 Explicit, 0 Non-Explicit, -1 All
        classifier_name = 'EXP_LEVEL1'
        # class_mapping_curr = dict([(k, v['ID']) for k, v in class_tree.iteritems()])
        class_mapping_curr = self.class_mapping
        load_model_file_classifier_current = '%s_%s.modelfile' % (load_model_file_basename, classifier_name)
        classifier_level1_exp = pickle.load(open(load_model_file_classifier_current, 'rb'))

        # Classifier: Non-Explicit, Level 1
        relation_type = 1  # 1 Explicit, 0 Non-Explicit, -1 All
        classifier_name = 'NONEXP_LEVEL1'
        # class_mapping_curr = dict([(k, v['ID']) for k, v in class_tree.iteritems()])
        class_mapping_curr = self.class_mapping
        load_model_file_classifier_current = '%s_%s.modelfile' % (load_model_file_basename, classifier_name)
        classifier_level1_nonexp = pickle.load(open(load_model_file_classifier_current, 'rb'))

        for i, relation_dict in enumerate(relation_dicts):
            # print relation_dict
            curr_features_vec = DiscourseSenseClassification_FeatureExtraction.extract_features_as_vector_from_single_record_v2_optimized( \
                relation_dict=relation_dict, \
                parse=parse, \
                word2vec_model=word2vec_model, \
                word2vec_index2word_set=word2vec_index2word_set, \
                deps_model=deps_model, \
                deps_vocabulary=set(deps_model._vocab),
            )

            if len(relation_dict['Connective']['TokenList']) > 0:
                relation_dict['Type'] = 'Explicit'
            else:
                relation_dict['Type'] = 'Implicit'

            # sense = valid_senses[random.randint(0, len(valid_senses) - 1)]

            if scale_features:
                curr_features_vec = scaler.transform([curr_features_vec])[0]

            if relation_dict['Type'] == 'Explicit':
                sense = classifier_level1_exp.predict([curr_features_vec])[0]
            else:
                sense = classifier_level1_nonexp.predict([curr_features_vec])[0]

            # print 'predicted sense:%s' % sense

            # TO DO classmaping id to original class mapping
            sense_original = class_mapping_id_to_origtext[sense]
            relation_dict['Sense'] = [sense_original]

            # set output data
            relation_dict['Arg1']['TokenList'] = \
                [x[2] for x in relation_dict['Arg1']['TokenList']]
            relation_dict['Arg2']['TokenList'] = \
                [x[2] for x in relation_dict['Arg2']['TokenList']]
            relation_dict['Connective']['TokenList'] = \
                [x[2] for x in relation_dict['Connective']['TokenList']]
            output.write(json.dumps(relation_dict) + '\n')

            if (i + 1) % 1000 == 0:
                print '%s of %s' % (i, len(relation_dicts))
                logging.info('%s of %s' % (i, len(relation_dicts)))
                print '%s features:%s' % (i, curr_features_vec)
        logging.info('output file written:%s' % output_file)


# Set logging info
logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s]: %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Enable file logging
logFileName = '%s/%s-%s.log' % ('logs', 'sup_parser_v1', '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now()))
fileHandler = logging.FileHandler(logFileName, 'wb')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

# Enable console logging
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# SAMPLE RUN:
# TRAIN:
# python sup_parser_v1.py en [dataset_folder_here] [model_folder_ghere] [output_dir_here] -run_name:sup_v1 -cmd:train -word2vec_model:""
#
#

if __name__ == '__main__':
    language = sys.argv[1]
    input_dataset = sys.argv[2]
    input_run = sys.argv[3]
    output_dir = sys.argv[4]
    if language == 'en':
        valid_senses = validator.EN_SENSES
    elif language == 'zh':
        valid_senses = validator.ZH_SENSES

    cmd = 'train'
    cmd = CommonUtilities.get_param_value("cmd", sys.argv, cmd)
    logging.info('cmd:%s' % cmd)

    # run name for output params
    run_name = ""
    run_name = CommonUtilities.get_param_value("run_name", sys.argv, run_name)
    if run_name != "":
        logging.info(('run_name:%s' % run_name))
    else:
        logging.error('Error: missing input file parameter - run_name')
        quit()

    # Perform scaling on the features
    scale_features = False
    scale_features = CommonUtilities.get_param_value_bool("scale_features", sys.argv, scale_features)
    logging.info('scale_features:{0}'.format(scale_features))

    # w2v/doc2vec params
    # word2vec word2vec_model file
    word2vec_model_file = ""  # "qatarliving\\qatarliving_size400_win10_mincnt10.word2vec.bin"
    word2vec_model_file = CommonUtilities.get_param_value("word2vec_model", sys.argv)
    logging.info('Word2Vec File:\n\t%s' % word2vec_model_file)
    # if word2vec_model_file == "":
    #    logging.error('Error: missing input file parameter - word2vec_model_file')
    #    quit()

    # wordclusters_mapping_file
    wordclusters_mapping_file = ""  # "qatarliving\\qatarliving_size400_win10_mincnt10.word2vec.bin"
    wordclusters_mapping_file = CommonUtilities.get_param_value("wordclusters_mapping_file", sys.argv)
    logging.info('wordclusters_mapping_file:\n\t%s' % wordclusters_mapping_file)

    doc2vec_model_file = ""  # "qatarliving\\qatarliving_size400_win10_mincnt10.word2vec.bin"
    doc2vec_model_file = CommonUtilities.get_param_value("doc2vec_model", sys.argv)
    if doc2vec_model_file != "":
        logging.info('Doc2Vec File:\n\t%s' % doc2vec_model_file)

    if doc2vec_model_file == '' and word2vec_model_file == '':
        logging.error('Error: missing input file parameter - either doc2vec_model_file or word2vec_model_file')
        quit()

    # use id for vector retrieval from doc2vec
    use_id_for_vector = False
    if sys.argv.count('-use_id_for_vector') > 0:
        use_id_for_vector = True
    logging.info('use_id_for_vector:{0}'.format(use_id_for_vector))

    # load word2vec word2vec_model as binary file
    word2vec_load_bin = False
    word2vec_load_bin = CommonUtilities.get_param_value_bool("word2vec_load_bin", sys.argv, word2vec_load_bin)
    logging.info('word2vec_load_bin:{0}'.format(word2vec_load_bin))

    # Brown clusters file
    brownclusters_file = ""
    brownclusters_file = CommonUtilities.get_param_value("brownclusters_file", sys.argv, brownclusters_file)
    logging.info('brownclusters_file:\n\t%s' % brownclusters_file)

    # dependency embeddings
    # word2vec word2vec_model file
    deps_model_file = ""  # "qatarliving\\qatarliving_size400_win10_mincnt10.word2vec.bin"
    deps_model_file = CommonUtilities.get_param_value("deps_model", sys.argv)
    logging.info('deps_model_file File:\n\t%s' % deps_model_file)

    deps_vocabulary = None
    deps_embeddings = None
    deps_model = None
    has_deps_embeddings = False

    if deps_model_file != "":
        has_deps_embeddings = True
        logging.info("Loading dependency embeddings from %s" % deps_model_file)
        deps_model = Embeddings.load(deps_model_file+".npy", deps_model_file+".vocab")
        logging.info("Deps Model loaded!")

        #deps_vocabulary = deps_model._vocab
        #deps_embeddings = deps_model._vecs


    # Load Models here
    is_doc2vec_model = False
    # load word2vec word2vec_model
    if doc2vec_model_file != '':
        model = Doc2Vec.load(doc2vec_model_file)
        is_doc2vec_model = True
    else:
        if word2vec_load_bin:
            model = Word2Vec.load_word2vec_format(word2vec_model_file, binary=True)  # use this for google vectors
        else:
            model = Word2Vec.load(word2vec_model_file)

    use_id_for_vector = use_id_for_vector and is_doc2vec_model

    word2vec_num_features = len(model.syn0[0])
    logging.info("Embeddings feature vectors length:%s" % word2vec_num_features)
    logging.info("Model syn0 len=%d" % (len(model.syn0)))

    # define classes
    class_mapping = dict([(val, idx) for idx, val in enumerate(valid_senses)])
    class_mapping_file = '%s/%s.classlabels' % (output_dir, run_name)
    CommonUtilities.write_dictionary_to_file(class_mapping, class_mapping_file)

    # RUN PARSER
    parser = DiscourseSenseClassifier_Sup_v2_Optimized_Hierarchical(valid_senses=valid_senses, input_run=input_run, input_dataset=input_dataset, \
                                    output_dir=output_dir, \
                                    input_params=None, input_features=None, \
                                    class_mapping=class_mapping)

    model_file_basename = '%s/%s_model_' % (input_run, run_name)
    scale_file_basename = '%s/%s_scalerange_' % (input_run, run_name)
    if cmd == 'train':
        logging.info('-----------TRAIN---------------------------------')
        parser.train_sense(input_dataset=input_dataset, word2vec_model=model, deps_model=deps_model,
                           save_model_file_basename=model_file_basename,
                           scale_features=scale_features, save_scale_file_basename=scale_file_basename)
    elif cmd == 'train-test':
        logging.debug(class_mapping)
        parser.train_sense(input_dataset=input_dataset, word2vec_model=model, deps_model=deps_model,
                           save_model_file_basename=model_file_basename,
                           scale_features=scale_features, save_scale_file_basename=scale_file_basename)
        logging.info('-------------------------------------------------------------')
        parser.classify_sense(input_dataset=input_dataset, word2vec_model=model, deps_model=deps_model,
                              load_model_file_basename=model_file_basename,
                              scale_features=scale_features, load_scale_file_basename=scale_file_basename)
    elif cmd == 'test':
        logging.info('-----------TEST----------------------------------')
        parser.classify_sense(input_dataset=input_dataset, word2vec_model=model, deps_model=deps_model,
                              load_model_file_basename=model_file_basename,
                              scale_features=scale_features, load_scale_file_basename=scale_file_basename)
    else:
        logging.error("command unknown: %s. Either -cmd:train or -cmd:test expected" % (cmd))
