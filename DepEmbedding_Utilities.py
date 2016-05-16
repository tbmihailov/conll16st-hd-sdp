from collections import Counter
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from gensim import models
import itertools
import numpy as np
import re
import sys
import os
import pickle

import logging

class DepEmbeddingUtilities(object):

    @staticmethod
    def load_word2vec_model(word2vec_model_filename, binary):
        """
        Loads word2vec model
        Args:
            word2vec_model_filename:Word2vec model file
            binary: If model is in binary format - used for Google News 300 distribution

        Returns:

        """
        model = models.Word2Vec.load_word2vec_format(word2vec_model_filename, binary=binary)

        return model

    @staticmethod
    def load_dependency_embeddings_model(model_filename):
        """
        Loads dependency embeddings from dep embeddings pickle file:
        Usage:
                model_dep = VocabEmbeddingUtilities.load_dependency_embeddings_model(vector_model_file)

                deps_vocabulary = model_dep['vocabulary']
                deps_embeddings = model_dep['embeddings']
        Args:
            model_filename: Dependency embeddings pickle model

        Returns:
            loaded dependency embeddings model:

        """
        model_dep = None
        with open(model_filename, 'rb') as f:
            model_dep = pickle.load(f)

        return model_dep

    @staticmethod
    def build_vocab_from_sentences(sentences):
        """
        Builds vocabulary from sentences. Words are ordered by count
        Args:
            sentences:
                List of lists of word tokens.
        Returns:
            Word types vocabulary
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

        return vocabulary

    @staticmethod
    def get_embeddings_for_vocab_from_model(vocabulary, embeddings_type, embeddings_model, embeddings_size):
        if embeddings_type == 'w2v':
            index2wordset = set(embeddings_model.index2word)  # performance optimization
            print ("Building embeddings...")
            vocab_size = len(vocabulary)
            print vocab_size
            embeddings = np.zeros((vocab_size, embeddings_size))
            for word in vocabulary:
                index = vocabulary[word]
                if word in index2wordset:
                    embeddings[index, :] = embeddings_model[word].reshape((1, embeddings_size))
                else:
                    # Init random embeddings vector
                    embeddings[index, :] = np.random.uniform(-0.23, 0.23, [1, embeddings_size])

            print ("Write vocab and embeddings data in a pickle...")

            try:
                # fp = open(output_pickle_file, 'wb')
                vocab_embeddings = {
                    'vocabulary': vocabulary,
                    'embeddings': embeddings
                }
                # pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
                # fp.close()
            except Exception as e:
                # print('Unable to save data to %s : %s' % (output_pickle_file, e))
                raise
        elif embeddings_type == 'random':
            vocab_size = len(vocabulary)
            embeddings = np.random.uniform(-1.0, 1.0, [vocab_size, embeddings_size])

            print ("Write data in a pickle...")
            # output_pickle_file = 'random.pickle'
            try:
                # fp = open(output_pickle_file, 'wb')
                vocab_embeddings = {
                    'vocabulary': vocabulary,
                    'embeddings': embeddings
                }
                # pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
                # fp.close()
            except Exception as e:
                # print('Unable to save data to %s : %s' % (output_pickle_file, e))
                raise
        elif embeddings_type == 'deps':
            print ("Loading deps embeddings_model...")

            vocabulary_deps = embeddings_model['vocabulary']
            embeddings_deps = embeddings_model['embeddings']

            print ("Building embeddings...")
            vocab_size = len(vocabulary)
            embeddings = np.zeros((vocab_size, embeddings_size))
            for word in vocabulary:
                index = vocabulary[word]
                try:
                    index_deps = vocabulary_deps[word]
                    embeddings[index, :] = embeddings_deps[index_deps, :]
                except KeyError:
                    embeddings[index, :] = np.random.uniform(-0.1, 0.1, [1, embeddings_size])

            print ("Write data in a pickle...")
            # output_pickle_file = 'deps.pickle'
            try:
                # fp = open(output_pickle_file, 'wb')
                vocab_embeddings = {
                    'vocabulary': vocabulary,
                    'embeddings': embeddings
                }
                # pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
                # fp.close()
            except Exception as e:
                # print('Unable to save data to %s : %s' % (output_pickle_file, e))
                raise
        else:
            raise Exception("vector_type must be in: %s" % ["w2v", "random", "deps"])

        return vocab_embeddings

    @staticmethod
    def build_vocab_and_embeddings_and_save_to_file(sentences, vectors_type, vector_model_file, vectors_size):
        """
        1. Builds vocabulary from sentences (array of word token arrays).
        2. Loads word embeddings from a specified word embeddings file and type
        3. Saves pickle file from object with Vocabulary and Embeddings props:
            {
                "Vocabulary" = vocab dictionary (word, idx)
                "Embeddings" = array indexed by vocabulary idx
            }
        Args:
            sentences:
            vectors_type:
            vector_model_file:
            vectors_size:
            output_pickle_file:

        Returns:

        """

        if vectors_type == 'w2v':
            print ("Loading w2v model...")
            try:
                model = VocabEmbeddingUtilities.load_word2vec_model(vector_model_file, True)
            except EOFError as e:
                print('Unable to load data from %s : %s' % (vector_model_file, e))
                raise
        elif vectors_type == 'random':
            # do nothing here - embveddings are generated randomly
            print "Embeddings will be generated randomly"
        elif vectors_type == 'deps':
            try:
                model = VocabEmbeddingUtilities.load_depembeddings_model(vector_model_file)
            except EOFError as e:
                print('Unable to load data from %s : %s' % (vector_model_file, e))
                raise
        else:
            raise Exception("vector_type must be in: %s" % ["w2v", "random", "deps"])

        vocab_embeddings = None
        print ("Building vocabulary...")
        # Build vocabulary
        vocabulary = VocabEmbeddingUtilities.build_vocab_from_sentences(sentences)

        vocab_embeddings = VocabEmbeddingUtilities.get_embeddings_for_vocab_from_model(vocabulary, vectors_type, model,
                                                                                       vectors_size)

        return vocab_embeddings
