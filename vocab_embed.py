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

def clean_str(string):
    """
    Tokenization/string cleaning; original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(epFile, deFile, dyFile):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    ep_examples = list(open(epFile,"rb").readlines())
    ep_examples = [s.decode('latin-1').strip() for s in ep_examples]

    de_examples = list(open(deFile,"rb").readlines())
    de_examples = [s.decode('latin-1').strip() for s in de_examples]

    dy_examples = list(open(dyFile,"rb").readlines())
    dy_examples = [s.decode('latin-1').strip() for s in dy_examples]

    # Split by words
    x_text = ep_examples + de_examples + dy_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    ep_labels = [[1, 0, 0] for _ in ep_examples]
    de_labels = [[0, 1, 0] for _ in de_examples]
    dy_labels = [[0, 0, 1] for _ in dy_examples]

    if not ep_labels:
        y = np.concatenate([de_labels, dy_labels], 0)
    if not de_labels:
        y = np.concatenate([ep_labels, dy_labels], 0)
    if not dy_labels:
        y = np.concatenate([ep_labels, de_labels], 0)
    else:
        y = np.concatenate([ep_labels, de_labels, dy_labels], 0)    

    return [x_text, y]

def pad_sentences(sentences, sentence_length, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sentence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab_and_embeddings(sentences, vector):
    """
    Recieves all sentences in MPQA and EPOS.
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    print ("Building vocabulary...")
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    if(vector == 'w2v'):
        print ("Loading w2v model...")
        model = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

        print ("Building embeddings...")
        vocab_size = len(vocabulary)
        embeddings = np.zeros((vocab_size, 300))
        for word in vocabulary:
        	index = vocabulary[word]
        	try:
        		embeddings[index, :] = model[word].reshape((1,300))
        	except KeyError:
        		embeddings[index, :] = np.random.uniform(-0.23, 0.23, [1,300])

        print ("Write data in a pickle...")
        pickle_file = 'w2v.pickle'
        try:
            fp = open(pickle_file, 'wb')
            save = {
                'vocabulary': vocabulary,
                'embeddings': embeddings
            }
            pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
            fp.close()
        except Exception as e:
            print ('Unable to save data to', pickle_file, ':', e)
            raise

    if (vector == 'random'):
        vocab_size = len(vocabulary)
        embeddings = np.random.uniform(-1.0, 1.0, [vocab_size, 300])

        print ("Write data in a pickle...")
        pickle_file = 'random.pickle'
        try:
            fp = open(pickle_file, 'wb')
            save = {
                'vocabulary': vocabulary,
                'embeddings': embeddings
            }
            pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
            fp.close()
        except Exception as e:
            print ('Unable to save data to', pickle_file, ':', e)
            raise

    if (vector == 'deps'):
        print ("Loading deps model...")
        pickle_file = 'model_deps.pickle'

        try:
            with open(pickle_file, 'rb') as f:
                save = pickle.load(f)
                vocabulary_deps = save['vocabulary']
                embeddings_deps = save['embeddings']
                del save
        except EOFError:
            return {'Unable to do something'}

        print ("Building embeddings...")
        vocab_size = len(vocabulary)
        embeddings = np.zeros((vocab_size, 300))
        for word in vocabulary:
            index = vocabulary[word]
            try:
                index_deps = vocabulary_deps[word]
                embeddings[index, :] = embeddings_deps[index_deps, :]
            except KeyError:
                embeddings[index, :] = np.random.uniform(-0.1, 0.1, [1,300])

        print ("Write data in a pickle...")
        pickle_file = 'deps.pickle'
        try:
            fp = open(pickle_file, 'wb')
            save = {
                'vocabulary': vocabulary,
                'embeddings': embeddings
            }
            pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
            fp.close()
        except Exception as e:
            print ('Unable to save data to', pickle_file, ':', e)
            raise


if __name__ == "__main__":

    ep_mpqa_file = '/home/mitarb/marasovic/CNN/MSC_data/MPQA/MPQA/cat/ep/ep.txt'
    de_mpqa_file = '/home/mitarb/marasovic/CNN/MSC_data/MPQA/MPQA/cat/de/de.txt'
    dy_mpqa_file = '/home/mitarb/marasovic/CNN/MSC_data/MPQA/MPQA/cat/dy/dy.txt'
    ep_epos_file = '/home/mitarb/marasovic/CNN/MSC_data/MPQA/EPOS/cat/ep/ep.txt'
    de_epos_file = '/home/mitarb/marasovic/CNN/MSC_data/MPQA/EPOS/cat/de/de.txt'
    dy_epos_file = '/home/mitarb/marasovic/CNN/MSC_data/MPQA/EPOS/cat/dy/dy.txt' 

    sentences_mpqa, labels_mpqa = load_data_and_labels(ep_mpqa_file, de_mpqa_file, dy_mpqa_file)
    sentence_length_mpqa = max(len(x) for x in sentences_mpqa)

    sentences_epos, labels_epos = load_data_and_labels(ep_epos_file, de_epos_file, dy_epos_file)
    sentence_length_epos = max(len(x) for x in sentences_epos)

    sentence_length = max(sentence_length_mpqa, sentence_length_epos)

    sentences_padded_mpqa = pad_sentences(sentences_mpqa, sentence_length)
    sentences_padded_epos = pad_sentences(sentences_epos, sentence_length)

    sentences_all = sentences_padded_mpqa + sentences_padded_epos

    vectors = ['w2v', 'random', 'deps']

    for vector in vectors:
    	build_vocab_and_embeddings(sentences_all, vector)