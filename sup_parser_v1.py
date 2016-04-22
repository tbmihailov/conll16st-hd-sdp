#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sample Discourse Relation Classifier Train

Train parser for suplementary evaluation

Train should take three arguments

	$inputDataset = the folder of the dataset to parse.
		The folder structure is the same as in the tar file
		$inputDataset/parses.json
		$inputDataset/relations-no-senses.json

	$inputRun = the folder that contains the model file or other resources

	$outputDir = the folder that the parser will output 'output.json' to

"""

import codecs
import json
import random
import sys
from datetime import datetime

import logging #word2vec logging

logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s]: %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Enable file logging
logFileName = '%s/%s-%s.log'%('logs', 'sup_parser_v1', '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now()))
fileHandler = logging.FileHandler(logFileName, 'wb')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

#Enable console logging
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# from sphinx.addnodes import index

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

class DiscourseParser_Sup_v1(object):
    """Sample discourse relation sense classifier
    
    This simply classifies each instance randomly. 
    """

    def __init__(self, valid_senses, input_run, input_dataset, output_dir, input_params, input_features,class_mapping
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
        postagged_sim = DiscourseParser_Sup_v1.calculate_postagged_similarity_from_taggeddata_and_tokens(
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
    def extract_features_as_vector_from_single_record(relation_dict, parse, word2vec_model, word2vec_index2word_set):
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

        # Connective embedding
        connective_words = [parse[doc_id]['sentences'][x[3]]['words'][x[4]][0] for x in
                            relation_dict['Connective']['TokenList']]
        connective_embedding = AverageVectorsUtilities.makeFeatureVec(connective_words, word2vec_model, w2v_num_feats,
                                                                word2vec_index2word_set)
        features.extend(connective_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, connective_embedding, 'W2V_CON_')

        # ARG 2
        arg2_tokens = [parse[doc_id]['sentences'][x[3]]['words'][x[4]] for x in relation_dict['Arg2']['TokenList']]
        arg2_words = [x[0] for x in arg2_tokens]
        # print 'arg2: %s' % arg2_words
        arg2_embedding = AverageVectorsUtilities.makeFeatureVec(arg2_words, word2vec_model, w2v_num_feats,
                                                                word2vec_index2word_set)
        features.extend(arg2_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, arg2_embedding, 'W2V_A2_')

        # Arg1 to Arg 2 cosine similarity
        arg1arg2_similarity = 0.00
        if len(arg1_words) > 0 and len(arg2_words) > 0:
            arg1arg2_similarity = spatial.distance.cosine(arg1_embedding, arg2_embedding)
        features.append(arg1arg2_similarity)

        # Calculate maximized similarities
        words1 = [x for x in arg1_words if x in word2vec_index2word_set]
        words2 = [x for x in arg1_words if x in word2vec_index2word_set]

        sim_avg_max = AverageVectorsUtilities.get_feature_vec_avg_aligned_sim(words1, words2, model,
                                                                              word2vec_num_features, word2vec_index2word_set)
        features.append(sim_avg_max)
        feat_key = "max_sim_aligned"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_max)

        sim_avg_top1 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, model,
                                                                                     word2vec_num_features,
                                                                                     word2vec_index2word_set, 1)
        features.append(sim_avg_top1)
        feat_key = "max_sim_avg_top1"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top1)

        sim_avg_top2 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, model,
                                                                                     word2vec_num_features,
                                                                                     word2vec_index2word_set, 2)
        features.append(sim_avg_top2)
        feat_key = "max_sim_avg_top2"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top2)

        sim_avg_top3 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, model,
                                                                                     word2vec_num_features,
                                                                                     word2vec_index2word_set, 3)
        features.append(sim_avg_top3)
        feat_key = "max_sim_avg_top3"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top3)

        sim_avg_top5 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, model,
                                                                                     word2vec_num_features,
                                                                                     word2vec_index2word_set, 5)
        features.append(sim_avg_top5)
        feat_key = "max_sim_avg_top5"
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, sim_avg_top5)

        # POS tags similarities
        postag_feats_vec, postag_feats_sparse = DiscourseParser_Sup_v1.get_postagged_sim_fetures(
            tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens, postagged_data_dict=parse,
            model=model, word2vec_num_features=word2vec_num_features,
            word2vec_index2word_set=word2vec_index2word_set)

        features.extend(postag_feats_vec)
        sparse_feats_dict.update(postag_feats_sparse)

        for i in range(0, len(features)):
            if math.isnan(features[i]):
                features[i] = 0.00

        return features #, sparse_feats_dict

    def train_sense(self, input_dataset, word2vec_model, save_model_file, scale_features, save_scale_file):
        class_mapping = self.class_mapping
        logging.debug(class_mapping)
        word2vec_index2word_set = set(word2vec_model.index2word)
        model_dir = self.input_run

        relation_file = '%s/relations.json' % input_dataset # with senses to train
        relation_dicts = [json.loads(x) for x in open(relation_file)]

        parse_file = '%s/parses.json' % input_dataset
        parse = json.load(codecs.open(parse_file, encoding='utf8'))

        random.seed(10)

        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

        train_x = []
        train_y = []
        for i, relation_dict in enumerate(relation_dicts):

            curr_features_vec = DiscourseParser_Sup_v1.extract_features_as_vector_from_single_record(\
                relation_dict=relation_dict,\
                parse=parse,\
                word2vec_model=word2vec_model,\
                word2vec_index2word_set=word2vec_index2word_set)

            if (i+1) % 1000 == 0:
                print '%s of %s' % (i, len(relation_dicts))
                logging.info('%s of %s' % (i, len(relation_dicts)))
                print '%s features:%s'%(i, curr_features_vec)

            curr_senseses = relation_dict['Sense'] # list of senses example: u'Sense': [u'Contingency.Cause.Reason']
            # logging.debug('%s - %s'%(i, curr_senseses))

            for curr_sense in curr_senseses:
                if curr_sense in class_mapping:
                    class_idx = class_mapping[curr_sense]
                    train_x.append(curr_features_vec)
                    train_y.append(class_idx)
                #else:
                #     logging.warn('Sense "%s" is not a valid class. Skip'%(curr_sense))


        scaler = preprocessing.MinMaxScaler(self.scale_range)
        if scale_features:
            logging.info('Scaling %s items with %s features..' % (len(train_x),len(train_x[0])))
            start = time.time()
            train_x = scaler.fit_transform(train_x)
            end = time.time()
            logging.info("Done in %s s" % (end - start))
            pickle.dump(scaler, open(save_scale_file, 'wb'))
            logging.info('Scale feats ranges saved to %s' % save_scale_file)
        else:
            logging.info("No scaling!")

        logging.info('Training with %s items' % len(train_x))
        start = time.time()
        clf.fit(train_x, train_y)
        end = time.time()
        logging.info("Done in %s s" % (end - start))

        pickle.dump(clf, open(save_model_file, 'wb'))
        logging.info('Model saved to %s' % save_model_file)

    def classify_sense(self, input_dataset, word2vec_model, load_model_file, scale_features, load_scale_file):
        output_dir = self.output_dir

        class_mapping = self.class_mapping
        class_mapping_id_to_origtext = dict([(value, key) for key,value in class_mapping.iteritems()])
        logging.debug('class_mapping_id_to_origtext:')
        logging.debug(class_mapping_id_to_origtext)

        word2vec_index2word_set = set(word2vec_model.index2word)

        relation_file = '%s/relations-no-senses.json' % input_dataset
        parse_file = '%s/parses.json' % input_dataset
        parse = json.load(codecs.open(parse_file, encoding='utf8'))

        relation_dicts = [json.loads(x) for x in open(relation_file)]

        output_file = '%s/output.json' % output_dir
        output = codecs.open(output_file, 'wb', encoding='utf8')

        clf = SVC()
        clf = pickle.load(open(load_model_file, 'rb'))

        if scale_features:
            # scaler = preprocessing.MinMaxScaler(self.scale_range)
            # scaler.transform(feats)
            scaler = pickle.load(open(load_scale_file, 'rb'))
            logger.info('Scaling is enabled!')
        else:
            logger.info('NO scaling!')

        for i, relation_dict in enumerate(relation_dicts):
            # print relation_dict
            curr_features_vec = DiscourseParser_Sup_v1.extract_features_as_vector_from_single_record( \
                relation_dict=relation_dict, \
                parse=parse, \
                word2vec_model=word2vec_model, \
                word2vec_index2word_set=word2vec_index2word_set)


            if len(relation_dict['Connective']['TokenList']) > 0:
                relation_dict['Type'] = 'Explicit'
            else:
                relation_dict['Type'] = 'Implicit'

            #sense = valid_senses[random.randint(0, len(valid_senses) - 1)]

            if scale_features:
                curr_features_vec = scaler.transform([curr_features_vec])[0]

            sense = clf.predict([curr_features_vec])[0]
            # print 'predicted sense:%s' % sense

            #TO DO classmaping id to original class mapping
            sense_original = class_mapping_id_to_origtext[sense]
            relation_dict['Sense'] = [sense_original]

            #set output data
            relation_dict['Arg1']['TokenList'] = \
                    [x[2] for x in relation_dict['Arg1']['TokenList']]
            relation_dict['Arg2']['TokenList'] = \
                    [x[2] for x in relation_dict['Arg2']['TokenList']]
            relation_dict['Connective']['TokenList'] = \
                    [x[2] for x in relation_dict['Connective']['TokenList']]
            output.write(json.dumps(relation_dict) + '\n')

            if (i+1) % 1000 == 0:
                print '%s of %s' % (i, len(relation_dicts))
                logging.info('%s of %s' % (i, len(relation_dicts)))
                print '%s features:%s' % (i, curr_features_vec)
        logging.info('output file written:%s' % output_file)

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
    logging.info('cmd:%s'%cmd)

    #run name for output params
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

    #w2v/doc2vec params
    # word2vec model file
    word2vec_model_file = ""  # "qatarliving\\qatarliving_size400_win10_mincnt10.word2vec.bin"
    word2vec_model_file = CommonUtilities.get_param_value("word2vec_model", sys.argv)
    if word2vec_model_file != "":
        logging.info('Word2Vec File:\n\t%s' % word2vec_model_file)
    # else:
    #    logging.error('Error: missing input file parameter - word2vec_model_file')
    #    quit()

    # wordclusters_mapping_file
    wordclusters_mapping_file = ""  # "qatarliving\\qatarliving_size400_win10_mincnt10.word2vec.bin"
    wordclusters_mapping_file = CommonUtilities.get_param_value("wordclusters_mapping_file", sys.argv)
    if wordclusters_mapping_file != "":
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

    # load word2vec model as binary file
    word2vec_load_bin = False
    word2vec_load_bin = CommonUtilities.get_param_value_bool("word2vec_load_bin", sys.argv, word2vec_load_bin)
    logging.info('word2vec_load_bin:{0}'.format(word2vec_load_bin))

    is_doc2vec_model = False
    # load word2vec model
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

    #RUN PARSER
    parser = DiscourseParser_Sup_v1(valid_senses=valid_senses, input_run=input_run, input_dataset=input_dataset,\
                                    output_dir=output_dir, \
                                    input_params=None, input_features=None,\
                                    class_mapping=class_mapping)

    model_file = '%s/%s.modelfile' % (input_run, run_name)
    scale_file = '%s/%s.scalerange' % (input_run, run_name)
    if cmd == 'train':
        logging.info('-----------TRAIN---------------------------------')
        parser.train_sense(input_dataset=input_dataset, word2vec_model=model, save_model_file=model_file,
                           scale_features=scale_features, save_scale_file=scale_file)
    elif cmd == 'train-test':
        logging.debug(class_mapping)
        parser.train_sense(input_dataset=input_dataset, word2vec_model=model, save_model_file=model_file,
                           scale_features=scale_features, save_scale_file=scale_file)
        logging.info('-------------------------------------------------------------')
        parser.classify_sense(input_dataset=input_dataset, word2vec_model=model, load_model_file=model_file,
                           scale_features=scale_features, load_scale_file=scale_file)
    elif cmd == 'test':
        logging.info('-----------TEST----------------------------------')
        parser.classify_sense(input_dataset=input_dataset, word2vec_model=model, load_model_file=model_file,
                           scale_features=scale_features, load_scale_file=scale_file)
    else:
        logging.error("command unknown: %s. Either -cmd:train or -cmd:test expected"%(cmd))


