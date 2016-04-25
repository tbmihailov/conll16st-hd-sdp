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
import random

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

def load_data_and_labels(sentences_file):
    lines = open(sentences_file,"r", encoding = 'latin-1').readlines()

    sentences_list = []
    labels_list = []

    for line in lines:
        line_split = line.split("\t")
        sentences_list.append(line_split[0])
        features = line_split[3].split(",")
        labels_list.append(features[len(features)-1].split('\n')[0])

    #sentences = [s.decode('latin-1').strip() for s in sentences_list]
    sentences = [clean_str(sent) for sent in sentences_list]
    sentences = [s.split(" ") for s in sentences]

    num_of_classes = 3 
    labels_set = {'ep': 0, 'de': 1, 'dy': 2}

    labels = []

    for label in labels_list:
        temp = [0]*num_of_classes
        index = labels_set[label]
        temp[index] = 1
        labels.append(temp)

    labels = np.asarray(labels)
    return [sentences, labels]

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


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

if __name__ == "__main__":

    filename = 'data_balanced_shuffled'
    f = open(filename, 'a')

    sentence_length = 79

    modal_verbs = ['can', 'could', 'may', 'must', 'should']

    for modal_verb in modal_verbs:
        f.write("Modal verb: " + modal_verb + "\n\n")

        pickle_file = 'w2v_' + modal_verb + "_new.pickle"

        print("Loading vocabulary...")

        try:
            with open(pickle_file, 'rb') as fp:
                save = pickle.load(fp)
                vocabulary = save['vocabulary']
                del save
        except EOFError:
            print ('Unable to do something')

        train_datasets = []
        train_labels = []
        test_datasets = []
        test_labels = []

        for k in range(5): 
            filename_train = "/home/mitarb/marasovic/CNN/MSC_data/MPQA/data_balanced/" + modal_verb + "/train/" + modal_verb + "_balance_classifier2_fold" + str(k+1) + ".txt"
            sentences_train, labels_train = load_data_and_labels(filename_train)

            filename_test = "/home/mitarb/marasovic/CNN/MSC_data/MPQA/data_balanced/" + modal_verb + "/test/" + modal_verb + "_balance_classifier2_fold" + str(k+1) + ".txt"
            sentences_test, labels_test = load_data_and_labels(filename_test)

            train_dataset, train_label = build_input_data(pad_sentences(sentences_train, sentence_length), labels_train, vocabulary)

            '''rng_state = np.random.get_state()
            np.random.shuffle(train_dataset)
            np.random.set_state(rng_state)
            np.random.shuffle(train_label)'''

            '''train = np.asarray(list(zip(train_dataset, train_label)))
            np.random.shuffle(train)
            train_dataset, train_label = zip(*train)
            train_dataset = np.asarray(train_dataset)
            train_label = np.asarray(train_label)'''

            test_dataset, test_label = build_input_data(pad_sentences(sentences_test, sentence_length), labels_test, vocabulary)

            train_datasets.append(train_dataset)
            train_labels.append(train_label)
            test_datasets.append(test_dataset)
            test_labels.append(test_label)

            f.write("Train data shape: " + str(np.shape(train_dataset)) + "\n\n")
            f.write("Test data shape: " + str(np.shape(test_dataset)) + "\n\n")

        vectors_list = ['w2v', 'deps', 'random']

        for vectors in vectors_list:
            pickle_file = vectors + '_' + modal_verb + "_new.pickle"

            print("Loading vocabulary and embeddings...")

            try:
                with open(pickle_file, 'rb') as fp:
                    save = pickle.load(fp)
                    vocabulary = save['vocabulary']
                    embeddings = save['embeddings']
                    del save
            except EOFError:
                print ('Unable to do something')

            print ("Write data in a pickle...")
            pickle_file = vectors + '_' + modal_verb +'_balanced.pickle'
            try:
                fp = open(pickle_file, 'wb')
                save = {
                    'train_datasets': train_datasets,
                    'train_labels': train_labels,
                    'test_datasets': test_datasets,
                    'test_labels': test_labels,
                    'vocabulary': vocabulary,
                    'embeddings': embeddings
                }
                pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
                fp.close()
            except Exception as e:
                print ('Unable to save data to', pickle_file, ':', e)
                raise

        f.write("="*10)
