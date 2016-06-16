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

import numpy

# from sklearn.svm import libsvm
from sklearn.svm import SVC
import copy
sys.path.append('~/semanticz')
from Word2Vec_AverageVectorsUtilities import AverageVectorsUtilities

import pickle

import const  # Constants support
import DiscourseSenseClassification_FeatureExtraction_v1
from DiscourseSenseClassification_FeatureExtraction_v1 import DiscourseSenseClassification_FeatureExtraction


from cnn_class_micro_static_extended import TextCNN_Ext

from VocabEmbedding_Utilities import VocabEmbeddingUtilities

const.padding_word = "<PAD/>"

import tensorflow as tf
import os

def pad_sentences(sentences, sentence_length, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    Credits: Ana Marasovic (marasovic@cl.uni-heidelberg.de)
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        new_sentence = pad_or_trim_sentence(sentence, sentence_length, padding_word)
        padded_sentences.append(new_sentence)

    return padded_sentences

def pad_or_trim_sentence(sentence, max_sentence_length, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    Credits: Ana Marasovic (marasovic@cl.uni-heidelberg.de)
    """
    if len(sentence)>max_sentence_length:
        new_sentence = sentence[:max_sentence_length]
    elif len(sentence) < max_sentence_length:
        num_padding = max_sentence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
    else:
        new_sentence = sentence[:]

    return new_sentence


class DiscourseSenseClassifier_Sup_v5_Hierarchical_CNN_Cross(object):
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

    @staticmethod
    def filter_items_train_classifier_and_save_model_logreg(classifier_name, class_mapping_curr, relation_type,
                                                                    train_x, train_y_txt,
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
        logging.info('======[%s] - filter_items_train_classifier_and_save_model_logreg======' % classifier_name)

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
        classifier_current = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                                 degree=3, gamma='auto', kernel='rbf',
                                 max_iter=-1, probability=False, random_state=None, shrinking=True,
                                 tol=0.001, verbose=False)
        print 'Classifier:\n%s' % classifier_current

        start = time.time()
        logging.info('Training with %s items...' % len(train_x_curr))
        classifier_current.fit(train_x_curr, train_y_curr)
        end = time.time()
        logging.info("Done in %s s" % (end - start))

        # Saving word2vec_model
        pickle.dump(classifier_current, open(save_model_file, 'wb'))
        logging.info('Model saved to %s' % save_model_file)

    @staticmethod
    def filter_items_train_classifier_and_save_model_cnn(classifier_name,
                                                         class_mapping_curr,
                                                         relation_type,
                                                         train_parsed_raw,
                                                         # train_y_txt,
                                                         # train_y_relation_types,
                                                         save_model_file,
                                                         vocabulary,
                                                         embeddings_model,
                                                         embeddings_type,
                                                         embeddings_size,
                                                         max_relation_length,
                                                         max_arg_length):

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
        logging.info('======[%s] - filter_items_train_classifier_and_save_model_cnn======' % classifier_name)

        vocab_embeddings = VocabEmbeddingUtilities.\
                            get_embeddings_for_vocab_from_model(
                                vocabulary=vocabulary,
                                embeddings_type=embeddings_type,
                                embeddings_model=embeddings_model,
                                embeddings_size=embeddings_size)

        # vocab_embeddings_file = '%s/vocab_embeddings_nn.pickle' % save_model_file
        vocab_embeddings_file = '%s_vocab_embeddings_nn.pickle' % save_model_file
        pickle.dump(vocab_embeddings, open(vocab_embeddings_file, 'wb'))
        logging.info('Vocab and embeddings saved to: %s' % vocab_embeddings_file)

        train_x_curr_s1 = []
        train_x_curr_s2 = []
        train_y_curr = []

        class_field = const.FIELD_LABEL_LEVEL2
        # Filtering items
        logging.info('Filtering %s items...' % len(train_parsed_raw))
        start = time.time()

        for i in range(0, len(train_parsed_raw)):
            if train_parsed_raw[i][class_field] in class_mapping_curr:  # and train_y_relation_types[i] == relation_type:
                # S2
                curr_train_tokens_s1 = train_parsed_raw[i][const.FIELD_ARG1]
                curr_train_tokens_s1 = pad_or_trim_sentence([x for x in curr_train_tokens_s1 if x in vocabulary],
                                                            max_arg_length, const.padding_word)
                curr_train_tokens_idx_s1 = [vocabulary[x] for x in curr_train_tokens_s1]
                train_x_curr_s1.append(curr_train_tokens_idx_s1)

                # S1
                curr_train_tokens_s2 = train_parsed_raw[i][const.FIELD_ARG2]
                curr_train_tokens_s2 = pad_or_trim_sentence([x for x in curr_train_tokens_s2 if x in vocabulary],
                                                            max_arg_length, const.padding_word)
                curr_train_tokens_idx_s2 = [vocabulary[x] for x in curr_train_tokens_s2]
                train_x_curr_s2.append(curr_train_tokens_idx_s2)

                train_y_curr.append(class_mapping_curr[train_parsed_raw[i][class_field]])

        end = time.time()
        logging.info("Done in %s s" % (end - start))

        # CNN CODE BELOW

        # Training
        # Classifier params
        l2_reg_lambda = 0.001
        num_epochs = 200 # default 100
        batch_size = 100 # default 50
        num_filters = 128
        dropout_keep_prob = 0.5

        regularisation = "y"

        logging.info(regularisation)
        if (regularisation == "n"):
            l2_reg_lambda = 0.0
            dropout_keep_prob = 1.0

        logging.info("Parameters:")
        logging.info("l2 regularisation: " + str(l2_reg_lambda))
        logging.info("Mini-batch size: " + str(batch_size))
        logging.info("Num_filters: " + str(num_filters))
        logging.info("Dropout keep prob: " + str(dropout_keep_prob))
        logging.info("Num of epochs: " + str(num_epochs))
        logging.info("Region sizes: 2,3,4")

        average_accuracy = 0.0

        split = 10
        total_train = len(train_x_curr_s1)

        train_to_take = int((total_train/split)*(split-1))
        train_dataset_s1 = numpy.array([x for x in train_x_curr_s1])
        train_dataset_s2 = numpy.array([x for x in train_x_curr_s2])

        train_label = numpy.array([[1 if (x-1)==i else 0 for i in range(15)] for x in train_y_curr])

        dev_dataset_s1 = numpy.array([x for x in train_x_curr_s1[train_to_take:]])
        dev_dataset_s2 = numpy.array([x for x in train_x_curr_s2[train_to_take:]])
        dev_label = numpy.array([[1 if (x - 1) == i else 0 for i in range(15)] for x in train_y_curr[train_to_take:]])

        logging.info("Split: Train - %s, Test - %s"%(len(train_dataset_s1), len(dev_dataset_s1)))

        shuffling = "n"
        # filter_sizes_1 = 3
        # filter_sizes_2 = 4
        # filter_sizes_3 = 5

        filter_sizes = [3, 4, 5]

        logging.info("Checking for inconsistent train data length...")
        sent_length = len(train_dataset_s1[0])
        for i in range(0, len(train_dataset_s1)):
            if len(train_dataset_s1[i]) != sent_length:
                logging.error("[%s]Wrong length: %s != %s"%(i, len(train_dataset_s1[i]), sent_length))
        print "Train sentence length: %s" % train_dataset_s1.shape[1]

        # cnn = TextCNN_Ext(train_dataset=train_dataset, train_labels=train_label, valid_dataset=test_dataset,
        #                   valid_labels=test_label, embeddings=vocab_embeddings['embeddings'], vocabulary=vocab_embeddings['vocabulary'],
        #                   l2_reg_lambda=l2_reg_lambda,
        #                   num_epochs=num_epochs, batch_size=batch_size, num_filters=num_filters,
        #                   filter_sizes_1=filter_sizes_1,
        #                   filter_sizes_2=filter_sizes_2, filter_sizes_3=filter_sizes_3,
        #                   dropout_keep_prob=dropout_keep_prob,
        #                   #  lexical=lexical,
        #                   shuffling=shuffling,
        #                   num_classes=len(class_mapping_curr))

        # logging.info(cnn.valid_accuracy)
        # logging.info("\n")
        # logging.info("Fold test accuracy: " + str(cnn.valid_accuracy))
        # average_accuracy = cnn.valid_accuracy


        from text_cnn_train import text_cnn_train_and_save_model_v2

        allow_soft_placement = True
        log_device_placement = False

        evaluate_every = len(train_dataset_s1)/batch_size
        checkpoint_every = 2*evaluate_every

        text_cnn_train_and_save_model_v2(x_train_s1=train_dataset_s1,
                                         x_train_s2=train_dataset_s2,
                                         y_train=train_label,
                                         x_dev_s1=dev_dataset_s1,
                                         x_dev_s2=dev_dataset_s2,
                                         y_dev=dev_label,
                                         out_dir=save_model_file,
                                         allow_soft_placement=allow_soft_placement,
                                         log_device_placement=log_device_placement,
                                         embeddings=vocab_embeddings['embeddings'],
                                         vocabulary=vocab_embeddings['vocabulary'],
                                         filter_sizes=filter_sizes,
                                         num_filters=num_filters,
                                         l2_reg_lambda=l2_reg_lambda,
                                         dropout_keep_prob=dropout_keep_prob,
                                         batch_size=batch_size,
                                         num_epochs=num_epochs,
                                         evaluate_every=evaluate_every,
                                         checkpoint_every=checkpoint_every,
                                         num_classes=len(class_mapping_curr))

        # average_accuracy = average_accuracy / 5.0
        # logging.info("Average accuracy on folds:")
        # logging.info(average_accuracy)
        # logging.info("Average accuracy on five folds: " + str(average_accuracy))
        # logging.info("=" * 60)
        # logging.info("=" * 60)


        # print 'Classifier:\n%s' % classifier_current
        #
        # start = time.time()
        # logging.info('Training with %s items...' % len(train_x_curr))
        # classifier_current.fit(train_x_curr, train_y_curr)
        # end = time.time()
        # logging.info("Done in %s s" % (end - start))
        #
        # # Saving word2vec_model
        # pickle.dump(classifier_current, open(save_model_file, 'wb'))
        # logging.info('Model saved to %s' % save_model_file)

    def train_sense(self, input_dataset, word2vec_model, save_model_file_basename, scale_features,
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

        # Build vocabulary
        # import os.path
        # if os.path.isfile(fname)

        vocab_tokens = {}
        max_id = 0
        max_arg_length = 0
        max_relation_length = 0

        # Add padding word to vocab
        vocab_tokens[const.padding_word] = max_id


        logging.info('=====EXTRACTING FEATURES======')

        train_items_with_raw_tokens_implicit = []
        logging.info('Extracting features from %s items..' % len(relation_dicts))

        for i, relation_dict in enumerate(relation_dicts):
            if (i + 1) % 1000 == 0:
                print '%s of %s' % (i, len(relation_dicts))
                logging.info('%s of %s' % (i, len(relation_dicts)))
                # print '%s features:%s' % (i, curr_features_implicit)

            curr_features_vec_explicit = []
            curr_features_implicit = {}
            if relation_dict['Type'] == 'Explicit':
                curr_features_vec_explicit = DiscourseSenseClassification_FeatureExtraction.extract_features_as_vector_from_single_record( \
                    relation_dict=relation_dict, \
                    parse=parse, \
                    word2vec_model=word2vec_model, \
                    word2vec_index2word_set=word2vec_index2word_set)

                curr_senses = relation_dict['Sense']  # list of senses example: u'Sense': [u'Contingency.Cause.Reason']
                # logging.debug('%s - %s'%(i, curr_senses))

                for curr_sense in curr_senses:
                    train_x.append(curr_features_vec_explicit)
                    # train_y.append(0) - not used

                    if '.' in curr_sense:
                        train_y_txt_level1.append(curr_sense.split('.')[0])
                    else:
                        train_y_txt_level1.append(curr_sense)
                    train_y_txt_level2.append(curr_sense)

                    if relation_dict['Type'] == 'Explicit':
                        train_y_relation_types.append(1)
                    else:
                        train_y_relation_types.append(0)

            else:
                curr_features_implicit = DiscourseSenseClassification_FeatureExtraction.extract_features_as_rawtokens_from_single_record( \
                    relation_dict=relation_dict, \
                    parse=parse)

                # determine max arg length for the dataset
                if len(curr_features_implicit[const.FIELD_ARG1]) > max_arg_length:
                    max_arg_length = len(curr_features_implicit[const.FIELD_ARG1])

                if len(curr_features_implicit[const.FIELD_ARG2]) > max_arg_length:
                    max_arg_length = len(curr_features_implicit[const.FIELD_ARG2])

                rel_len = len(curr_features_implicit[const.FIELD_ARG1]) + len(curr_features_implicit[const.FIELD_ARG1]) + \
                          len(curr_features_implicit[const.FIELD_CONNECTIVE])

                # determine max relation length
                if rel_len > max_relation_length:
                    max_relation_length = rel_len

                # Build vocab and calc params
                for token in curr_features_implicit[const.FIELD_ARG1]:
                    if not token in vocab_tokens:
                        max_id = max_id + 1
                        vocab_tokens[token] = max_id

                for token in curr_features_implicit[const.FIELD_ARG2]:
                    if not token in vocab_tokens:
                        max_id = max_id + 1
                        vocab_tokens[token] = max_id

                for token in curr_features_implicit[const.FIELD_CONNECTIVE]:
                    if not token in vocab_tokens:
                        max_id = max_id + 1
                        vocab_tokens[token] = max_id

                curr_senses = relation_dict['Sense']  # list of senses example: u'Sense': [u'Contingency.Cause.Reason']
                # logging.debug('%s - %s'%(i, curr_senses))

                # deal with multiple labels
                for curr_sense in curr_senses:
                    copied_features_implicit = copy.deepcopy(curr_features_implicit)

                    copied_features_implicit[const.FIELD_LABEL_LEVEL1] = curr_sense.split('.')[0] if '.' in curr_sense else curr_sense
                    copied_features_implicit[const.FIELD_LABEL_LEVEL2] = curr_sense

                    copied_features_implicit[const.FIELD_REL_TYPE] = 1 if relation_dict['Type'] == 'Explicit' else 0

                    train_items_with_raw_tokens_implicit.append(copied_features_implicit)

        # logging.info('Vocab size: %s' % vocab_tokens)

        vocab_and_stat_file = '%s_vocabandstat.pickle' % (save_model_file_basename)

        data_vocab_and_stat = {
            'vocabulary': vocab_tokens,
            'max_relation_length': max_relation_length,
            'max_arg_length': max_arg_length,
        }
        pickle.dump(data_vocab_and_stat, open(vocab_and_stat_file,'wb'))
        logging.info('Vocab saved to: %s' % vocab_and_stat_file)
        #for k, v in vocab_tokens.iteritems():
        #    print "%s - %s" % (k, v)

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

        ###########################
        ### FILTER AND TRAIN ######
        ###########################

        # Classifier: Non-Explicit, Level 1
        relation_type = 0  # 1 Explicit, 0 Non-Explicit, -1 All
        classifier_name = 'NONEXP_LEVEL1_CNN'
        # class_mapping_curr = dict([(k, v['ID']) for k, v in class_tree.iteritems()])
        class_mapping_curr = class_mapping_flat
        save_model_file_classifier_current = '%s_%s.tensorflow' % (save_model_file_basename, classifier_name)


        DiscourseSenseClassifier_Sup_v5_Hierarchical_CNN_Cross\
            .filter_items_train_classifier_and_save_model_cnn(classifier_name=classifier_name,
                                                              class_mapping_curr=class_mapping_curr,
                                                              relation_type=relation_type,
                                                              train_parsed_raw=train_items_with_raw_tokens_implicit,
                                                              # train_y_txt=train_y_txt_level2,
                                                              # train_y_relation_types=train_y_relation_types,
                                                              save_model_file=save_model_file_classifier_current,
                                                              vocabulary=vocab_tokens,
                                                              max_relation_length = max_relation_length,
                                                              max_arg_length=max_arg_length,
                                                              embeddings_model=word2vec_model,
                                                              embeddings_type="w2v",
                                                              embeddings_size=word2vec_num_features
                                                              )

        # Classifier: Explicit, Level 1
        relation_type = 1  # 1 Explicit, 0 Non-Explicit, -1 All
        classifier_name = 'EXP_LEVEL1'
        # class_mapping_curr = dict([(k, v['ID']) for k, v in class_tree.iteritems()])
        class_mapping_curr = class_mapping_flat
        save_model_file_classifier_current = '%s_%s.modelfile' % (save_model_file_basename, classifier_name)

        DiscourseSenseClassifier_Sup_v5_Hierarchical_CNN_Cross\
            .filter_items_train_classifier_and_save_model_logreg(classifier_name=classifier_name,
                                                                class_mapping_curr=class_mapping_curr,
                                                                relation_type=relation_type,
                                                                train_x=train_x,
                                                                train_y_txt=train_y_txt_level2,
                                                                train_y_relation_types=train_y_relation_types,
                                                                save_model_file=save_model_file_classifier_current)

    def classify_sense(self, input_dataset, word2vec_model, load_model_file_basename, scale_features,
                       load_scale_file_basename, hierachical_classifier=False):
        output_dir = self.output_dir

        class_mapping = self.class_mapping
        class_mapping_id_to_origtext = dict([(value, key) for key, value in class_mapping.iteritems()])
        logging.debug('class_mapping_id_to_origtext:')
        logging.debug(class_mapping_id_to_origtext)

        word2vec_index2word_set = set(word2vec_model.index2word)

        relation_file = '%s/relations-no-senses.json' % input_dataset
        relation_dicts = [json.loads(x) for x in open(relation_file)]

        parse_file = '%s/parses.json' % input_dataset
        parse = json.load(codecs.open(parse_file, encoding='utf8'))

        relation_file_gold = '%s/relations.json' % input_dataset
        has_gold = False
        relation_dicts_gold = []
        relations_y_gold_implicit = []
        if os.path.isfile(relation_file_gold):
            print "GOLD DATA LOADED"
            has_gold = True
            relation_dicts_gold = [json.loads(x) for x in open(relation_file_gold)]

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

        #vocab and stat
        vocab_and_stat_file = '%s_vocabandstat.pickle' % (load_model_file_basename)
        logging.info('Loading vocab and stat from: %s' % vocab_and_stat_file)
        data_vocab_and_stat = pickle.load(open(vocab_and_stat_file, 'rb'))
        # vocabulary = data_vocab_and_stat['vocabulary']
        max_relation_length = data_vocab_and_stat['max_relation_length']
        max_arg_length = data_vocab_and_stat['max_arg_length']

        # Classifier: Non-Explicit, Level 1
        relation_type = 1  # 1 Explicit, 0 Non-Explicit, -1 All
        classifier_name = 'NONEXP_LEVEL1_CNN'
        # class_mapping_curr = dict([(k, v['ID']) for k, v in class_tree.iteritems()])
        class_mapping_curr = self.class_mapping
        load_model_file_classifier_current = '%s_%s.tensorflow' % (load_model_file_basename, classifier_name)
        # classifier_level1_nonexp = pickle.load(open(load_model_file_classifier_current, 'rb'))

        # vocab_embeddings_file = '%s/vocab_embeddings_nn.pickle' % save_model_file
        vocab_embeddings_file = '%s_vocab_embeddings_nn.pickle' % load_model_file_classifier_current
        vocab_embeddings = pickle.load(open(vocab_embeddings_file, 'rb'))
        logging.info('Vocab and embeddings loaded from: %s' % vocab_embeddings_file)
        vocabulary = vocab_embeddings['vocabulary']
        embeddings = vocab_embeddings['embeddings']


        curr_features_implicit_list = []
        implicit_relation_objects_list = []  # used for updating sense estimation result

        for i, relation_dict in enumerate(relation_dicts):
            if len(relation_dict['Connective']['TokenList']) > 0:
                relation_dict['Type'] = 'Explicit'
            else:
                relation_dict['Type'] = 'Implicit'

            # print relation_dict
            if relation_dict['Type'] == 'Explicit':
                curr_features_vec = DiscourseSenseClassification_FeatureExtraction.extract_features_as_vector_from_single_record( \
                    relation_dict=relation_dict, \
                    parse=parse, \
                    word2vec_model=word2vec_model, \
                    word2vec_index2word_set=word2vec_index2word_set)

                if scale_features:
                    curr_features_vec = scaler.transform([curr_features_vec])[0]
                    sense = classifier_level1_exp.predict([curr_features_vec])[0]

                sense = classifier_level1_exp.predict([curr_features_vec])[0]
                # print 'predicted sense:%s' % sense
                # TO DO classmaping id to original class mapping
                sense_original = class_mapping_id_to_origtext[sense]
                relation_dict['Sense'] = [sense_original]

            else:
                # sense = classifier_level1_nonexp.predict([curr_features_vec])[0]
                curr_features_implicit = DiscourseSenseClassification_FeatureExtraction\
                    .extract_features_as_rawtokens_from_single_record( \
                    relation_dict = relation_dict, \
                    parse = parse)

                curr_features_implicit_list.append(curr_features_implicit)
                implicit_relation_objects_list.append(relation_dict)

                random.seed(42)
                from random import randint
                if has_gold:
                    relation_dict_g = relation_dicts_gold[i]

                    if relation_dict_g['Sense'][0] in class_mapping:
                        relations_y_gold_implicit.append(class_mapping[relation_dict_g['Sense'][0]])
                    else:

                        relations_y_gold_implicit.append(randint(0, len(class_mapping)-1))

            # set output data
            relation_dict['Arg1']['TokenList'] = \
                [x[2] for x in relation_dict['Arg1']['TokenList']]
            relation_dict['Arg2']['TokenList'] = \
                [x[2] for x in relation_dict['Arg2']['TokenList']]
            relation_dict['Connective']['TokenList'] = \
                [x[2] for x in relation_dict['Connective']['TokenList']]

            # output.write(json.dumps(relation_dict) + '\n')

            if (i + 1) % 100 == 0:
                print '%s of %s' % (i, len(relation_dicts))
                logging.info('%s of %s' % (i, len(relation_dicts)))
                print '%s features:%s' % (i, curr_features_vec)

        logging.info('Explicit relations predicted!')

        logging.info('Predicting non-explicit relations...')

        # NON EXPLICIT - EXVALUATION CNN

        from text_cnn_eval import text_cnn_load_model_and_eval_v2

        checkpoint_dir = os.path.abspath(os.path.join(load_model_file_classifier_current, "checkpoints"))
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        allow_soft_placement = True
        log_device_placement = False

        train_x_curr_s1 = []
        train_x_curr_s2 = []
        # Filtering items
        logging.info('Preprocessing %s test items...' % len(curr_features_implicit_list))
        start = time.time()

        for i in range(0, len(curr_features_implicit_list)):
            #S2
            curr_train_tokens_s1 = curr_features_implicit_list[i][const.FIELD_ARG1]
            curr_train_tokens_s1 = pad_or_trim_sentence([x for x in curr_train_tokens_s1 if x in vocabulary],
                                                        max_arg_length, const.padding_word)
            curr_train_tokens_idx_s1 = [vocabulary[x] for x in curr_train_tokens_s1]
            train_x_curr_s1.append(curr_train_tokens_idx_s1)

            #S1
            curr_train_tokens_s2 = curr_features_implicit_list[i][const.FIELD_ARG2]
            curr_train_tokens_s2 = pad_or_trim_sentence([x for x in curr_train_tokens_s2 if x in vocabulary],
                                                        max_arg_length, const.padding_word)
            curr_train_tokens_idx_s2 = [vocabulary[x] for x in curr_train_tokens_s2]
            train_x_curr_s2.append(curr_train_tokens_idx_s2)




        print "Embeddings[%s] : %s"%(100, vocab_embeddings['embeddings'][100])
        print "train_x_curr: %s"%len(train_x_curr_s1)
        predictions_y = text_cnn_load_model_and_eval_v2(x_test_s1=train_x_curr_s1,
                                                        x_test_s2=train_x_curr_s2,
                                                        checkpoint_file=checkpoint_file,
                                                        allow_soft_placement=allow_soft_placement,
                                                        log_device_placement=log_device_placement,
                                                        embeddings=embeddings)

        print "Classes:"
        sorted_class_mappings = sorted([(k, v) for k, v in class_mapping_id_to_origtext.iteritems()])
        print sorted_class_mappings

        print "Predictions:"
        predictions_y = [min(len(class_mapping_id_to_origtext), x+1) for x in predictions_y]
        print predictions_y

        print "Classes:"
        for (idx,lbl) in sorted_class_mappings:
            print "%s - %s" % (idx, lbl)


        #Confusion matrix

        if has_gold:
            print "Implicit confusion matrix"
            import sklearn.metrics as skm

            conf_matrix = skm.confusion_matrix(relations_y_gold_implicit, predictions_y)
            print conf_matrix

            print skm.accuracy_score(relations_y_gold_implicit, predictions_y)

        # Print accuracy
        # correct_predictions = float(sum(predictions_y == y_test))
        # print("Total number of test examples: {}".format(len(y_test)))
        # print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

        # set predicted labels
        #for i, relation_dict in enumerate(implicit_relation_objects_list):
        print "predictions_y cnt:%s" % len(predictions_y)
        print "implicit_relation_objects_list cnt:%s" % len(implicit_relation_objects_list)
        for i in range(0, len(predictions_y)):
            # label_binary = predictions_y[i]
            # label = next(obj for idx,obj in enumerate(label_binary) if obj == 1)+1
            label = predictions_y[i]
            if len(implicit_relation_objects_list[i]['Sense']) > 0:
                logging.error("%s - already filled - %s" % (i, implicit_relation_objects_list[i]['Type']))
            else:
                implicit_relation_objects_list[i]['Sense'] = [class_mapping_id_to_origtext[label]]

        # export results
        for i, relation_dict in enumerate(relation_dicts):
            output.write(json.dumps(relation_dict) + '\n')

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

    # Load Models here
    is_doc2vec_model = False
    # load word2vec word2vec_model
    if doc2vec_model_file != '':
        word2vec_model = Doc2Vec.load(doc2vec_model_file)
        is_doc2vec_model = True
    else:
        if word2vec_load_bin:
            word2vec_model = Word2Vec.load_word2vec_format(word2vec_model_file, binary=True)  # use this for google vectors
        else:
            word2vec_model = Word2Vec.load(word2vec_model_file)

    use_id_for_vector = use_id_for_vector and is_doc2vec_model

    word2vec_num_features = len(word2vec_model.syn0[0])
    logging.info("Embeddings feature vectors length:%s" % word2vec_num_features)
    logging.info("Model syn0 len=%d" % (len(word2vec_model.syn0)))

    # define classes
    class_mapping = dict([(val, idx) for idx, val in enumerate(valid_senses)])
    class_mapping_file = '%s/%s.classlabels' % (output_dir, run_name)
    CommonUtilities.write_dictionary_to_file(class_mapping, class_mapping_file)

    # RUN PARSER
    parser = DiscourseSenseClassifier_Sup_v5_Hierarchical_CNN_Cross(valid_senses=valid_senses,
                                                                    input_run=input_run,
                                                                    input_dataset=input_dataset,
                                                                    output_dir=output_dir,
                                                                    input_params=None,
                                                                    input_features=None,
                                                                    class_mapping=class_mapping)

    model_file_basename = '%s/%s_model_' % (input_run, run_name)
    scale_file_basename = '%s/%s_scalerange_' % (input_run, run_name)
    if cmd == 'train':
        logging.info('-----------TRAIN---------------------------------')
        parser.train_sense(input_dataset=input_dataset, word2vec_model=word2vec_model,
                           save_model_file_basename=model_file_basename,
                           scale_features=scale_features, save_scale_file_basename=scale_file_basename)
    elif cmd == 'train-test':
        logging.debug(class_mapping)
        parser.train_sense(input_dataset=input_dataset, word2vec_model=word2vec_model,
                           save_model_file_basename=model_file_basename,
                           scale_features=scale_features, save_scale_file_basename=scale_file_basename)
        logging.info('-------------------------------------------------------------')
        parser.classify_sense(input_dataset=input_dataset, word2vec_model=word2vec_model,
                              load_model_file_basename=model_file_basename,
                              scale_features=scale_features, load_scale_file_basename=scale_file_basename)
    elif cmd == 'test':
        logging.info('-----------TEST----------------------------------')
        parser.classify_sense(input_dataset=input_dataset, word2vec_model=word2vec_model,
                              load_model_file_basename=model_file_basename,
                              scale_features=scale_features, load_scale_file_basename=scale_file_basename)
    else:
        logging.error("command unknown: %s. Either -cmd:train or -cmd:test expected" % (cmd))
