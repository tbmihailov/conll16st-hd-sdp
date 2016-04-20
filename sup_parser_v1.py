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

sys.path.append('/semanticz')


class DiscourseParser_Sup_v1(object):
    """Sample discourse relation sense classifier
    
    This simply classifies each instance randomly. 
    """

    def __init__(self, valid_senses, input_run, input_dataset, output_dir, input_params, input_features):
        self.valid_senses = valid_senses
        self.input_run = input_run
        self.input_dataset = input_dataset
        self.output_dir = output_dir
        self.input_params = input_params
        self.input_features = input_features

        pass

    @staticmethod
    def get_word_token(parse_obj, doc_id, sent_id, word_id):
        return parse_obj[doc_id]['sentences'][sent_id]['words'][word_id]

    @staticmethod
    def extract_features_as_vector_from_single_record(relation_dict, parse):
        features = []

        # FEATURES EXTRACTION HERE
        arg1_tokenList = [x[2] for x in relation_dict['Arg1']['TokenList']]
        arg2_tokenList = [x[2] for x in relation_dict['Arg2']['TokenList']]
        connective_tokenList = [x[2] for x in relation_dict['Connective']['TokenList']]
        has_connective = 1 if len(connective_tokenList) > 0 else 0

        return features

    def train_sense(self, input_dataset, word2vec_model):
        word2vec_index2word_set = set(word2vec_model.index2word)


        relation_file = '%s/relations.json' % input_dataset # with senses to train
        relation_dicts = [json.loads(x) for x in open(relation_file)]

        parse_file = '%s/parses.json' % input_dataset
        parse = json.load(codecs.open(parse_file, encoding='utf8'))

        random.seed(10)
        for i, relation_dict in enumerate(relation_dicts):
            doc_id = relation_dict['DocID']

            curr_features_vec = DiscourseParser_Sup_v1.extract_features_as_vector_from_single_record(realtion_dict, parse)

            senses_lst = relation_dict['Sense'] # list of senses example: u'Sense': [u'Contingency.Cause.Reason']

            # train the classifier here or export features..



    def classify_sense(self):
        input_dataset = self.input_dataset
        output_dir = self.output_dir

        relation_file = '%s/relations-no-senses.json' % input_dataset
        parse_file = '%s/parses.json' % input_dataset
        parse = json.load(codecs.open(parse_file, encoding='utf8'))

        relation_dicts = [json.loads(x) for x in open(relation_file)]

        output = codecs.open('%s/output.json' % output_dir, 'wb', encoding='utf8')
        random.seed(10)
        for i, relation_dict in enumerate(relation_dicts):
            sense = valid_senses[random.randint(0, len(valid_senses)-1)]
            relation_dict['Sense'] = [sense]
            relation_dict['Arg1']['TokenList'] = \
                    [x[2] for x in relation_dict['Arg1']['TokenList']]
            relation_dict['Arg2']['TokenList'] = \
                    [x[2] for x in relation_dict['Arg2']['TokenList']]
            relation_dict['Connective']['TokenList'] = \
                    [x[2] for x in relation_dict['Connective']['TokenList']]
            if len(relation_dict['Connective']['TokenList']) > 0:
                relation_dict['Type'] = 'Explicit'
            else:
                relation_dict['Type'] = 'Implicit'
            output.write(json.dumps(relation_dict) + '\n')

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
    print 'cmd:%s'%cmd

    #run name for output params
    run_name = ""
    run_name = CommonUtilities.get_param_value("run_name", sys.argv, run_name)
    if run_name!="":
        print('run_name:%s' % run_name)
    else:
        print('Error: missing input file parameter - run_name')
        quit()

    #w2v/doc2vec params
    # word2vec model file
    word2vec_model_file = ""  # "qatarliving\\qatarliving_size400_win10_mincnt10.word2vec.bin"
    word2vec_model_file = CommonUtilities.get_param_value("word2vec_model", sys.argv)
    if word2vec_model_file != "":
        print('Word2Vec File:\n\t%s' % word2vec_model_file)
    # else:
    #    print('Error: missing input file parameter - word2vec_model_file')
    #    quit()

    # wordclusters_mapping_file
    wordclusters_mapping_file = ""  # "qatarliving\\qatarliving_size400_win10_mincnt10.word2vec.bin"
    wordclusters_mapping_file = CommonUtilities.get_param_value("wordclusters_mapping_file", sys.argv)
    if wordclusters_mapping_file != "":
        print('wordclusters_mapping_file:\n\t%s' % wordclusters_mapping_file)

    doc2vec_model_file = ""  # "qatarliving\\qatarliving_size400_win10_mincnt10.word2vec.bin"
    doc2vec_model_file = CommonUtilities.get_param_value("doc2vec_model", sys.argv)
    if doc2vec_model_file != "":
        print('Doc2Vec File:\n\t%s' % doc2vec_model_file)

    if doc2vec_model_file == '' and word2vec_model_file == '':
        print('Error: missing input file parameter - either doc2vec_model_file or word2vec_model_file')
        quit()

    # use id for vector retrieval from doc2vec
    use_id_for_vector = False
    if sys.argv.count('-use_id_for_vector') > 0:
        use_id_for_vector = True
    print('use_id_for_vector:{0}'.format(use_id_for_vector))

    # load word2vec model as binary file
    word2vec_load_bin = False
    word2vec_load_bin = CommonUtilities.get_param_value_bool("word2vec_load_bin", sys.argv, word2vec_load_bin)
    print('word2vec_load_bin:{0}'.format(word2vec_load_bin))

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
    print "Embeddings feature vectors length:%s" % word2vec_num_features
    print "Model syn0 len=%d" % (len(model.syn0))

    # define clases
    relevance_to_2class_mapping = dict([(val, idx) for idx, val in enumerate(valid_senses)])
    relevance_to_2class_mapping_file = '%s/%s.classlabels' % (output_dir, run_name)
    CommonUtilities.write_dictionary_to_file(relevance_to_2class_mapping, relevance_to_2class_mapping_file)

    #RUN PARSER
    parser = DiscourseParser_Sup_v1(valid_senses=valid_senses, input_run=input_run, input_dataset=input_dataset,\
                                    output_dir=output_dir)

    if cmd == 'train':
        parser.train_sense()
    elif cmd == 'test':
        parser.classify_sense()
    else:
        print "command unknown: %s. Either -cmd:train or -cmd:test expected"%(cmd)


