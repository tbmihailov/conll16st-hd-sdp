from sklearn import cross_validation
from gensim import models
import numpy as np
import pickle
import sys

modal_verb = sys.argv[1]
vectors = sys.argv[2]
tuning = sys.argv[3]
lexical = sys.argv[4]
filename = sys.argv[5]
regularisation = sys.argv[6]
shuffling = sys.argv[7]
filter_sizes_1 = int(sys.argv[8])
filter_sizes_2 = int(sys.argv[9])
filter_sizes_3 = int(sys.argv[10])


print (shuffling)

if (tuning == 'static'):
    from cnn_class_micro_static import TextCNN
# if (tuning == 'tuned'):
#    from cnn_class_micro_tuned import TextCNN

pickle_file = vectors + '_' + modal_verb + '_balanced.pickle'

with open(pickle_file, 'rb') as fp:
    save = pickle.load(fp)
    train_datasets = save['train_datasets']
    train_labels = save['train_labels']
    test_datasets = save['test_datasets']
    test_labels = save['test_labels']
    vocabulary = save['vocabulary']
    embeddings = save['embeddings']
    del save

'''train_datasets = train_datasets_inner
train_labels = train_labels_inner
test_datasets = test_datasets_inner
test_labels = test_labels_inner'''

modal_verbs = ['can', 'could', 'may', 'must', 'should']
if (lexical == "notlex"):
    for modal in modal_verbs:
        index = vocabulary[modal]
        embeddings[index, :] = np.zeros((1, 300))

# filename = 'impact_of_word_vectors'
fw = open(filename, 'a')

fw.write("Modal verb: " + modal_verb + "\n\n")
fw.write("Tuning: " + tuning + "\n\n")
fw.write("Vectors: " + vectors + "\n\n")
fw.write("Ignore (notlex) or not (lex) modal verb in sentence: " + lexical + "\n\n")

l2_reg_lambda = 0.001
num_steps = 1001
batch_size = 50
num_filters = 100
dropout_keep_prob = 0.5

print (regularisation)
if (regularisation == "n"):
    l2_reg_lambda = 0.0
    dropout_keep_prob = 1.0

print (l2_reg_lambda)
fw.write("Parameters:" + "\n\n")
fw.write("l2 regularisation: " + str(l2_reg_lambda) + "\n\n")
fw.write("Mini-batch size: " + str(batch_size) + "\n\n")
fw.write("Num_filters: " + str(num_filters) + "\n\n")
fw.write("Dropout keep prob: " + str(dropout_keep_prob) + "\n\n")
fw.write("Num of iter: " + str(num_steps) + "\n\n")
fw.write("Region sizes: 2,3,4" + "\n\n")

average_accuracy = 0.0

for k in range(5):
    train_dataset = train_datasets[k]
    train_label = train_labels[k]
    test_dataset = test_datasets[k]
    test_label = test_labels[k]
    print ("Fold " + str(k + 1) + ":\n\n")
    fw.write("Fold " + str(k + 1) + ":\n\n")
    cnn = TextCNN(train_dataset=train_dataset, train_labels=train_label, valid_dataset=test_dataset,
                  valid_labels=test_label, embeddings=embeddings, vocabulary=vocabulary, l2_reg_lambda=l2_reg_lambda,
                  num_steps=num_steps, batch_size=batch_size, num_filters=num_filters, filter_sizes_1=filter_sizes_1,
                  filter_sizes_2=filter_sizes_2, filter_sizes_3=filter_sizes_3, dropout_keep_prob=dropout_keep_prob,
                  lexical=lexical, shuffling=shuffling)
    print (cnn.valid_accuracy)
    print ("\n")
    fw.write("Fold test accuracy: " + str(cnn.valid_accuracy) + "\n\n")
    average_accuracy += cnn.valid_accuracy

average_accuracy = average_accuracy / 5.0
print ("Average accuracy on folds:")
print (average_accuracy)
fw.write("Average accuracy on five folds: " + str(average_accuracy) + "\n\n")
print("=" * 100)
fw.write("=" * 100 + "\n\n")
fw.close()
