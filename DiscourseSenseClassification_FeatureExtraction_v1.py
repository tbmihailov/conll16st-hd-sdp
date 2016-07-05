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

import logging #word2vec logging


from sklearn import preprocessing

import validator
from Common_Utilities import CommonUtilities

import gensim
from gensim import corpora, models, similarities # used for word2vec
from gensim.models.word2vec import Word2Vec # used for word2vec
from gensim.models.doc2vec import Doc2Vec#used for doc2vec

import time # used for performance measuring
import math

from scipy import spatial # used for similarity calculation
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Phrases

from gensim import corpora # for dictionary
from gensim.models import LdaModel

# from sklearn.svm import libsvm
from sklearn.svm import SVC

sys.path.append('~/semanticz')
from Word2Vec_AverageVectorsUtilities import AverageVectorsUtilities

import pickle

import const

# Constants
const.FIELD_ARG1 = 'Arg1'
const.FIELD_ARG2 = 'Arg2'
const.FIELD_CONNECTIVE = 'Connective'
const.FIELD_LABEL_LEVEL1 = 'Lbl_Lvl1'
const.FIELD_LABEL_LEVEL2 = 'Lbl_Lvl2'
const.FIELD_REL_TYPE = 'Type'

class DiscourseSenseClassification_FeatureExtraction(object):
    """Discourse relation sense classifier feature extration
    """
    CONNECTIVES_FREELY_OMISSIBLE= [
        'accordingly',
        'as a result',
        'because',
        'by comparison',
        'by contrast',
        'consequently',
        'for example',
        'for instance',
        'furthermore',
        'in fact',
        'in other words',
        'in particular',
        'in short',
        'indeed',
        'previously',
        'rather',
        'so',
        'specifically',
        'therefore'
    ]

    CONNECTIVES = ['if',
     'rather',
     'in contrast',
     'conversely',
     'neither nor',
     'meantime',
     'therefore',
     'while',
     'in other words',
     'so',
     'as well',
     'after',
     'afterward',
     'once',
     'upon',
     'so that',
     'nonetheless',
     'by contrast',
     'particularly',
     'likewise',
     'now that',
     'indeed',
     'further',
     'before',
     'previously',
     'in addition',
     'also',
     'separately',
     'besides',
     'until',
     'thereafter',
     'thereby',
     'hence',
     'in fact',
     'much as',
     'when',
     'for',
     'since',
     'or',
     'consequently',
     'similarly',
     'in the end',
     'thus',
     'in particular',
     'simultaneously',
     'nevertheless',
     'on the other hand',
     'whereas',
     'lest',
     'furthermore',
     'if and when',
     'in sum',
     'although',
     'regardless',
     'moreover',
     'on the contrary',
     'overall',
     'alternatively',
     'as long as',
     'then',
     'plus',
     'before and after',
     'meanwhile',
     'by comparison',
     'when and if',
     'yet',
     'in the mean time',
     'as soon as',
     'accordingly',
     'on the one hand on the other hand',
     'by then',
     'earlier',
     'however',
     'as if',
     'except',
     'though',
     'later',
     'next',
     'in turn',
     'still',
     'either or',
     'unless',
     'else',
     'as',
     'as a result',
     'insofar as',
     'otherwise',
     'instead',
     'because',
     'for instance',
     'finally',
     'till',
     'in short',
     'but',
     'if then',
     'nor',
     'ultimately',
     'specifically',
     'as though',
     'as an alternative',
     'and',
     'for example',
     'additionally']

    @staticmethod
    def get_connectives_emeddings(connectives, model, vocab_set, num_feats, split_words=True):

        if split_words:
            connectives_split = [c.split(' ') for c in connectives]
        else:
            connectives_split = [c for c in connectives]

        connectives_embedd = []

        for conn_words in connectives_split:
            conn_embedding = AverageVectorsUtilities.makeFeatureVec(conn_words, model, num_feats,
                                                                    vocab_set)
            connectives_embedd.append(conn_embedding)

        return connectives_embedd

    @staticmethod
    def calc_sim_singleembedd_to_embeddlist(single_embedding, embedding_list):
        sim_list = []

        for i in range(0, len(embedding_list)):
            sim = spatial.distance.cosine(single_embedding, embedding_list[i])
            sim_list.append(sim)

        return sim_list

    @staticmethod
    def get_connectives_emeddings(connectives, model, vocab_set, num_feats):

        connectives_split = [c.split(' ') for c in connectives]

        connectives_embedd = []

        for conn_words in connectives_split:
            conn_embedding = AverageVectorsUtilities.makeFeatureVec(conn_words, model, num_feats,
                                                                    vocab_set)
            connectives_embedd.append(conn_embedding)

        return connectives_embedd

    @staticmethod
    def get_word_token(parse_obj, doc_id, sent_id, word_id):
        return parse_obj[doc_id]['sentences'][sent_id]['words'][word_id]


    @staticmethod
    def calculate_postagged_similarity_from_taggeddata_and_tokens(text1_tokens_in_vocab,
                                                                  text2_tokens_in_vocab,
                                                                  model,
                                                                  tag_type_start_1,
                                                                  tag_type_start_2):
        res_sim = 0.00

        text1_words_in_model = [x[0] for x in text1_tokens_in_vocab if x[1]['PartOfSpeech'].startswith(tag_type_start_1)]
        text2_words_in_model = [x[0] for x in text2_tokens_in_vocab if x[1]['PartOfSpeech'].startswith(tag_type_start_2)]

        if len(text1_words_in_model) > 0 and len(text2_words_in_model) > 0:
            res_sim = model.n_similarity(text1_words_in_model, text2_words_in_model)

        return res_sim


    @staticmethod
    def get_postagged_sim_fetures(tokens_data_text1, tokens_data_text2, postagged_data_dict,
                                  model,
                                  word2vec_num_features,
                                  word2vec_index2word_set
                                  ):
        input_data_wordvectors = []
        input_data_sparse_features = {}

        tokens_in_vocab_1 = [x for x in tokens_data_text1 if x[0] in word2vec_index2word_set]
        tokens_in_vocab_2 = [x for x in tokens_data_text2 if x[0] in word2vec_index2word_set]

        # similarity for  tag type
        tag_type_start_1 = 'NN'
        tag_type_start_2 = 'NN'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'J'
        tag_type_start_2 = 'J'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'VB'
        tag_type_start_2 = 'VB'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'RB'
        tag_type_start_2 = 'RB'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'DT'
        tag_type_start_2 = 'DT'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'PR'
        tag_type_start_2 = 'PR'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'NN'
        tag_type_start_2 = 'J'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'J'
        tag_type_start_2 = 'NN'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'RB'
        tag_type_start_2 = 'VB'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'VB'
        tag_type_start_2 = 'RB'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'PR'
        tag_type_start_2 = 'NN'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # similarity for  tag type
        tag_type_start_1 = 'NN'
        tag_type_start_2 = 'PR'
        postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
            text1_tokens_in_vocab=tokens_in_vocab_1,
            text2_tokens_in_vocab=tokens_in_vocab_2,
            model=model,
            tag_type_start_1=tag_type_start_1,
            tag_type_start_2=tag_type_start_2)

        input_data_wordvectors.append(postagged_sim)
        input_data_sparse_features[
            'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
            postagged_sim

        # Additional features
        include_modal = True
        if include_modal:
            # similarity for  tag type
            tag_type_start_1 = 'MD'
            tag_type_start_2 = 'VB'
            postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
                text1_tokens_in_vocab=tokens_in_vocab_1,
                text2_tokens_in_vocab=tokens_in_vocab_2,
                model=model,
                tag_type_start_1=tag_type_start_1,
                tag_type_start_2=tag_type_start_2)

            input_data_wordvectors.append(postagged_sim)
            input_data_sparse_features[
                'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
                postagged_sim

            # similarity for  tag type
            tag_type_start_1 = 'VB'
            tag_type_start_2 = 'MD'
            postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
                text1_tokens_in_vocab=tokens_in_vocab_1,
                text2_tokens_in_vocab=tokens_in_vocab_2,
                model=model,
                tag_type_start_1=tag_type_start_1,
                tag_type_start_2=tag_type_start_2)

            input_data_wordvectors.append(postagged_sim)
            input_data_sparse_features[
                'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
                postagged_sim

            # similarity for  tag type
            tag_type_start_1 = ''
            tag_type_start_2 = 'MD'
            postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
                text1_tokens_in_vocab=tokens_in_vocab_1,
                text2_tokens_in_vocab=tokens_in_vocab_2,
                model=model,
                tag_type_start_1=tag_type_start_1,
                tag_type_start_2=tag_type_start_2)

            input_data_wordvectors.append(postagged_sim)
            input_data_sparse_features[
                'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
                postagged_sim

            # similarity for  tag type
            tag_type_start_1 = 'MD'
            tag_type_start_2 = ''
            postagged_sim = DiscourseSenseClassification_FeatureExtraction.calculate_postagged_similarity_from_taggeddata_and_tokens(
                text1_tokens_in_vocab=tokens_in_vocab_1,
                text2_tokens_in_vocab=tokens_in_vocab_2,
                model=model,
                tag_type_start_1=tag_type_start_1,
                tag_type_start_2=tag_type_start_2)

            input_data_wordvectors.append(postagged_sim)
            input_data_sparse_features[
                'sim_pos_arg1_%s_arg2_%s' % (tag_type_start_1, 'ALL' if tag_type_start_2 == '' else tag_type_start_2)] = \
                postagged_sim

        return input_data_wordvectors, input_data_sparse_features


    @staticmethod
    def extract_features_as_vector_from_single_record_v1(relation_dict, parse, word2vec_model, word2vec_index2word_set,
                                                         deps_model, deps_vocabulary):
        features = []
        sparse_feats_dict = {}

        deps_num_feats = deps_model.shape[1]
        w2v_num_feats = len(word2vec_model.syn0[0])

        # FEATURE EXTRACTION HERE
        doc_id = relation_dict['DocID']
        # print doc_id
        connective_tokenlist = [x[2] for x in relation_dict['Connective']['TokenList']]

        has_connective = 1 if len(connective_tokenlist) > 0 else 0
        features.append(has_connective)
        feat_key = "has_connective"
        if has_connective == 1:
            CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, has_connective)

        # print 'relation_dict:'
        # print relation_dict['Arg1']['TokenList']

        # ARG 1
        arg1_tokens = [parse[doc_id]['sentences'][x[3]]['words'][x[4]] for x in relation_dict['Arg1']['TokenList']]
        arg1_words = [x[0] for x in arg1_tokens]

        # print 'arg1: %s' % arg1_words
        arg1_embedding = AverageVectorsUtilities.makeFeatureVec(arg1_words, word2vec_model, w2v_num_feats,
                                                                word2vec_index2word_set)
        features.extend(arg1_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg1_embedding, 'W2V_A1_')

        # arg1 deps embeddings
        arg1_embedding_deps = AverageVectorsUtilities.makeFeatureVec(arg1_words, deps_model, deps_num_feats,
                                                                deps_vocabulary)

        features.extend(arg1_embedding_deps)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg1_embedding_deps, 'DEPS_A1_')


        # connective embedding
        connective_words = [parse[doc_id]['sentences'][x[3]]['words'][x[4]][0] for x in
                            relation_dict['Connective']['TokenList']]
        connective_embedding = AverageVectorsUtilities.makeFeatureVec(connective_words, word2vec_model, w2v_num_feats,
                                                                      word2vec_index2word_set)
        features.extend(connective_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, connective_embedding, 'W2V_CON_')

        # Connective DEPS embveddings
        connective_embedding_deps = AverageVectorsUtilities.makeFeatureVec(connective_words, deps_model, deps_num_feats,
                                                                      deps_vocabulary)

        features.extend(connective_embedding_deps)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, connective_embedding_deps, 'DEPS_CON_')

        # ARG 2
        arg2_tokens = [parse[doc_id]['sentences'][x[3]]['words'][x[4]] for x in relation_dict['Arg2']['TokenList']]
        arg2_words = [x[0] for x in arg2_tokens]
        # print 'arg2: %s' % arg2_words
        arg2_embedding = AverageVectorsUtilities.makeFeatureVec(arg2_words, word2vec_model, w2v_num_feats,
                                                                word2vec_index2word_set)
        features.extend(arg2_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg2_embedding, 'W2V_A2_')

        # arg2 deps embeddings
        arg2_embedding_deps = AverageVectorsUtilities.makeFeatureVec(arg2_words, deps_model, deps_num_feats,
                                                                     deps_vocabulary)

        features.extend(arg2_embedding_deps)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg2_embedding_deps, 'DEPS_A2_')

        # Arg1 to Arg 2 cosine similarity
        arg1arg2_similarity = 0.00
        if len(arg1_words) > 0 and len(arg2_words) > 0:
            arg1arg2_similarity = spatial.distance.cosine(arg1_embedding, arg2_embedding)
        features.append(arg1arg2_similarity)

        # Calculate maximized similarities
        words1 = [x for x in arg1_words if x in word2vec_index2word_set]
        words2 = [x for x in arg1_words if x in word2vec_index2word_set]

        sim_avg_max = AverageVectorsUtilities.get_feature_vec_avg_aligned_sim(words1, words2, word2vec_model,
                                                                              w2v_num_feats,
                                                                              word2vec_index2word_set)
        features.append(sim_avg_max)
        feat_key = "max_sim_aligned"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_max)

        sim_avg_top1 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 1)
        features.append(sim_avg_top1)
        feat_key = "max_sim_avg_top1"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top1)

        sim_avg_top2 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 2)
        features.append(sim_avg_top2)
        feat_key = "max_sim_avg_top2"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top2)

        sim_avg_top3 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 3)
        features.append(sim_avg_top3)
        feat_key = "max_sim_avg_top3"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top3)

        sim_avg_top5 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 5)
        features.append(sim_avg_top5)
        feat_key = "max_sim_avg_top5"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top5)

        # POS tags similarities
        postag_feats_vec, postag_feats_sparse = DiscourseSenseClassification_FeatureExtraction.get_postagged_sim_fetures(
            tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens, postagged_data_dict=parse,
            model=word2vec_model, word2vec_num_features=w2v_num_feats,
            word2vec_index2word_set=word2vec_index2word_set)

        # print postag_feats_vec

        features.extend(postag_feats_vec)
        sparse_feats_dict.update(postag_feats_sparse)

        for i in range(0, len(features)):
            if math.isnan(features[i]):
                features[i] = 0.00

        return features  # , sparse_feats_dict

    @staticmethod
    def extract_features_as_vector_from_single_record_v2_optimized(relation_dict, parse, word2vec_model, word2vec_index2word_set,
                                                         deps_model, deps_vocabulary, use_connective_sim=True, return_sparse_feats = False):
        features = []
        sparse_feats_dict = {}

        deps_num_feats = deps_model.shape[1]
        w2v_num_feats = len(word2vec_model.syn0[0])

        # FEATURE EXTRACTION HERE
        doc_id = relation_dict['DocID']
        # print doc_id
        connective_tokenlist = [x[2] for x in relation_dict['Connective']['TokenList']]

        has_connective = 1 if len(connective_tokenlist) > 0 else 0
        features.append(has_connective)
        feat_key = "has_connective"
        if has_connective == 1:
            CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, has_connective)

        # print 'relation_dict:'
        # print relation_dict['Arg1']['TokenList']

        # ARG 1
        arg1_tokens = [parse[doc_id]['sentences'][x[3]]['words'][x[4]] for x in relation_dict['Arg1']['TokenList']]
        arg1_words = [x[0] for x in arg1_tokens]

        # print 'arg1: %s' % arg1_words
        arg1_embedding = AverageVectorsUtilities.makeFeatureVec(arg1_words, word2vec_model, w2v_num_feats,
                                                                word2vec_index2word_set)
        features.extend(arg1_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg1_embedding, 'W2V_A1_')

        # arg1 deps embeddings
        arg1_embedding_deps = AverageVectorsUtilities.makeFeatureVec(arg1_words, deps_model, deps_num_feats,
                                                                     deps_vocabulary)

        features.extend(arg1_embedding_deps)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg1_embedding_deps, 'DEPS_A1_')

        # connective embedding
        connective_words = [parse[doc_id]['sentences'][x[3]]['words'][x[4]][0] for x in
                            relation_dict['Connective']['TokenList']]
        connective_embedding = AverageVectorsUtilities.makeFeatureVec(connective_words, word2vec_model, w2v_num_feats,
                                                                      word2vec_index2word_set)
        features.extend(connective_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, connective_embedding, 'W2V_CON_')

        # Connective DEPS embveddings
        connective_embedding_deps = AverageVectorsUtilities.makeFeatureVec(connective_words, deps_model, deps_num_feats,
                                                                           deps_vocabulary)

        features.extend(connective_embedding_deps)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, connective_embedding_deps, 'DEPS_CON_')

        # ARG 2
        arg2_tokens = [parse[doc_id]['sentences'][x[3]]['words'][x[4]] for x in relation_dict['Arg2']['TokenList']]
        arg2_words = [x[0] for x in arg2_tokens]
        # print 'arg2: %s' % arg2_words
        arg2_embedding = AverageVectorsUtilities.makeFeatureVec(arg2_words, word2vec_model, w2v_num_feats,
                                                                word2vec_index2word_set)
        features.extend(arg2_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg2_embedding, 'W2V_A2_')

        # arg2 deps embeddings
        arg2_embedding_deps = AverageVectorsUtilities.makeFeatureVec(arg2_words, deps_model, deps_num_feats,
                                                                     deps_vocabulary)

        features.extend(arg2_embedding_deps)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg2_embedding_deps, 'DEPS_A2_')

        # Arg1 to Arg 2 cosine similarity
        arg1arg2_similarity = 0.00
        if len(arg1_words) > 0 and len(arg2_words) > 0:
            arg1arg2_similarity = spatial.distance.cosine(arg1_embedding, arg2_embedding)
        features.append(arg1arg2_similarity)

        # Calculate maximized similarities
        words1 = [x for x in arg1_words if x in word2vec_index2word_set]
        words2 = [x for x in arg1_words if x in word2vec_index2word_set]

        sim_avg_max = AverageVectorsUtilities.get_feature_vec_avg_aligned_sim(words1, words2, word2vec_model,
                                                                              w2v_num_feats,
                                                                              word2vec_index2word_set)
        features.append(sim_avg_max)
        feat_key = "max_sim_aligned"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_max)

        sim_avg_top1 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 1)
        features.append(sim_avg_top1)
        feat_key = "max_sim_avg_top1"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top1)

        sim_avg_top2 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 2)
        features.append(sim_avg_top2)
        feat_key = "max_sim_avg_top2"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top2)

        sim_avg_top3 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 3)
        features.append(sim_avg_top3)
        feat_key = "max_sim_avg_top3"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top3)

        sim_avg_top5 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 5)
        features.append(sim_avg_top5)
        feat_key = "max_sim_avg_top5"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top5)

        # POS tags similarities
        postag_feats_vec, postag_feats_sparse = DiscourseSenseClassification_FeatureExtraction.get_postagged_sim_fetures(
            tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens, postagged_data_dict=parse,
            model=word2vec_model, word2vec_num_features=w2v_num_feats,
            word2vec_index2word_set=word2vec_index2word_set)

        features.extend(postag_feats_vec)
        sparse_feats_dict.update(postag_feats_sparse)

        for i in range(0, len(features)):
            if math.isnan(features[i]):
                features[i] = 0.00

        if return_sparse_feats:
            return features, sparse_feats_dict
        else:
            return features


    @staticmethod
    def extract_features_as_vector_from_single_record(relation_dict, parse, word2vec_model, word2vec_index2word_set,
                                                      connective_embedd_list=None,
                                                      include_connective_features=True,
                                                      return_sparse_feats=False):
        features = []
        sparse_feats_dict = {}

        w2v_num_feats = len(word2vec_model.syn0[0])
        # FEATURE EXTRACTION HERE
        doc_id = relation_dict['DocID']
        # print doc_id
        connective_tokenlist = [x[2] for x in relation_dict['Connective']['TokenList']]

        has_connective = 1 if len(connective_tokenlist) > 0 else 0
        features.append(has_connective)
        feat_key = "has_connective"
        #if has_connective == 1:
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, has_connective)

        # print 'relation_dict:'
        # print relation_dict['Arg1']['TokenList']

        # ARG 1
        arg1_tokens = [parse[doc_id]['sentences'][x[3]]['words'][x[4]] for x in relation_dict['Arg1']['TokenList']]
        arg1_words = [x[0] for x in arg1_tokens]

        # print 'arg1: %s' % arg1_words
        arg1_embedding = AverageVectorsUtilities.makeFeatureVec(arg1_words, word2vec_model, w2v_num_feats,
                                                                word2vec_index2word_set)
        features.extend(arg1_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg1_embedding, 'W2V_A1_')
        sparse_feats_dict.update(vec_feats)

        # Connective embedding
        if include_connective_features:
            connective_words = [parse[doc_id]['sentences'][x[3]]['words'][x[4]][0] for x in
                                relation_dict['Connective']['TokenList']]
            connective_embedding = AverageVectorsUtilities.makeFeatureVec(connective_words, word2vec_model, w2v_num_feats,
                                                                          word2vec_index2word_set)
            features.extend(connective_embedding)
            vec_feats = {}
            CommonUtilities.append_features_with_vectors(vec_feats, connective_embedding, 'W2V_CON_')
            sparse_feats_dict.update(vec_feats)

        # ARG 2
        arg2_tokens = [parse[doc_id]['sentences'][x[3]]['words'][x[4]] for x in relation_dict['Arg2']['TokenList']]
        arg2_words = [x[0] for x in arg2_tokens]
        # print 'arg2: %s' % arg2_words
        arg2_embedding = AverageVectorsUtilities.makeFeatureVec(arg2_words, word2vec_model, w2v_num_feats,
                                                                word2vec_index2word_set)
        features.extend(arg2_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg2_embedding, 'W2V_A2_')
        sparse_feats_dict.update(vec_feats)

        # Arg1 to Arg 2 cosine similarity
        arg1arg2_similarity = 0.00
        if len(arg1_words) > 0 and len(arg2_words) > 0:
            arg1arg2_similarity = spatial.distance.cosine(arg1_embedding, arg2_embedding)
        features.append(arg1arg2_similarity)
        feat_key = "sim_arg1arg2"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity)

        # Calculate maximized similarities
        words1 = [x for x in arg1_words if x in word2vec_index2word_set]
        words2 = [x for x in arg1_words if x in word2vec_index2word_set]

        sim_avg_max = AverageVectorsUtilities.get_feature_vec_avg_aligned_sim(words1, words2, word2vec_model,
                                                                              w2v_num_feats,
                                                                              word2vec_index2word_set)
        features.append(sim_avg_max)
        feat_key = "max_sim_aligned"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_max)

        sim_avg_top1 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 1)
        features.append(sim_avg_top1)
        feat_key = "max_sim_avg_top1"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top1)

        sim_avg_top2 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 2)
        features.append(sim_avg_top2)
        feat_key = "max_sim_avg_top2"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top2)

        sim_avg_top3 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 3)
        features.append(sim_avg_top3)
        feat_key = "max_sim_avg_top3"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top3)

        sim_avg_top5 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, word2vec_model,
                                                                                     w2v_num_feats,
                                                                                     word2vec_index2word_set, 5)
        features.append(sim_avg_top5)
        feat_key = "max_sim_avg_top5"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top5)

        # POS tags similarities
        postag_feats_vec, postag_feats_sparse = DiscourseSenseClassification_FeatureExtraction.get_postagged_sim_fetures(
            tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens, postagged_data_dict=parse,
            model=word2vec_model, word2vec_num_features=w2v_num_feats,
            word2vec_index2word_set=word2vec_index2word_set)
        # print postag_feats_sparse

        features.extend(postag_feats_vec)
        sparse_feats_dict.update(postag_feats_sparse)

        # calculate connectives similarity
        if connective_embedd_list is not None:
            arg1arg2_avg = (arg1_embedding+arg2_embedding)/2
            connective_sims = DiscourseSenseClassification_FeatureExtraction.\
                                calc_sim_singleembedd_to_embeddlist(arg1arg2_avg, connective_embedd_list)

            # print connective_sims
            features.extend(connective_sims)
            vec_feats = {}
            CommonUtilities.append_features_with_vectors(vec_feats, connective_sims, 'A1A2_CONNSIMS_')
            sparse_feats_dict.update(vec_feats)

        #else:
        #    # Extend with zeros for explicit
        #    features.extend([0 for x in DiscourseSenseClassification_FeatureExtraction.CONNECTIVES])


        # Set None to zero
        for i in range(0, len(features)):
            if math.isnan(features[i]):
                features[i] = 0.00

        # Set None to zero
        for k in sparse_feats_dict.iterkeys():
            if math.isnan(sparse_feats_dict[k]):
                sparse_feats_dict[k] = 0.00

        if return_sparse_feats:
            return features, sparse_feats_dict
        else:
            return features



    @staticmethod
    def extract_features_as_rawtokens_from_single_record(relation_dict, parse):
        features = {}

        # FEATURE EXTRACTION HERE
        doc_id = relation_dict['DocID']
        # print doc_id
        connective_tokenlist = [x[2] for x in relation_dict['Connective']['TokenList']]

        has_connective = 1 if len(connective_tokenlist) > 0 else 0
        # features.append(has_connective)
        feat_key = "has_connective"

        features['HasConnective'] = has_connective

        # print 'relation_dict:'
        # print relation_dict['Arg1']['TokenList']

        # ARG 1
        arg1_tokens = [parse[doc_id]['sentences'][x[3]]['words'][x[4]] for x in relation_dict['Arg1']['TokenList']]
        arg1_words = [x[0] for x in arg1_tokens]

        features[const.FIELD_ARG1] = arg1_words

        # Connective embedding
        connective_words = [parse[doc_id]['sentences'][x[3]]['words'][x[4]][0] for x in
                            relation_dict['Connective']['TokenList']]

        features[const.FIELD_CONNECTIVE] = connective_words

        # ARG 2
        arg2_tokens = [parse[doc_id]['sentences'][x[3]]['words'][x[4]] for x in relation_dict['Arg2']['TokenList']]
        arg2_words = [x[0] for x in arg2_tokens]
        # print 'arg2: %s' % arg2_words

        features[const.FIELD_ARG2] = arg2_words

        return features
