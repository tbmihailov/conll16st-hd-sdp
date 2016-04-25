#average feature vectors
import numpy as np
import gensim
from gensim import corpora, models, similarities # used for word2vec
from gensim.models.word2vec import Word2Vec # used for word2vec
import sys

from scipy import spatial # used for similarity calculation

class AverageVectorsUtilities(object):
    @staticmethod
    def makeFeatureVec(words, model, num_features, index2word_set):
        #function to average all words vectors in a given paragraph
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0

        #list containing names of words in the vocabulary
        #index2word_set = set(word2vec_model.index2word) this is moved as input param for performance reasons
        for word in words:
            if word in index2word_set:
                nwords = nwords+1
                featureVec = np.add(featureVec, model[word])

        if(nwords>0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    #get average similarity between every word from word1 with closes word in word2
    @staticmethod
    def get_feature_vec_avg_aligned_sim(words1, words2, model, num_features, index2word_set):
        #function to average all words vectors in a given paragraph
        nwords = 0

        #list containing names of words in the vocabulary
        #index2word_set = set(word2vec_model.index2word) this is moved as input param for performance reasons
        aligned_sim=0.00
        sim_sum=0.00
        for word1 in words1:
            if word1 in index2word_set:
                nwords = nwords+1
                best_sim=0.0000
                for word2 in words2:
                    if word2 in index2word_set:
                        sim = model.similarity(word1, word2)
                        if sim > best_sim:
                            best_sim=sim
                sim_sum+=best_sim

        if(nwords>0):
            aligned_sim = sim_sum/nwords
        return aligned_sim

    #get average similarity between every word from word1 with closes word in word2
    @staticmethod
    def get_question_vec_to_top_words_avg_sim(words1, words2, model, num_features, index2word_set, top_num_words):
        #function to average all words vectors in a given paragraph
        if(len(words1)==0 or len(words2)==0):
            return 0.00

        nwords = 0 #all words checked
        avg_sim=0.00 #avg similarity to be returned

        top_sims=[]

        words1_in_model = words1
        for word2 in words2:
            if word2 in index2word_set:
                nwords = nwords+1
                sim = model.n_similarity(words1_in_model, [word2])
                top_sims.append(sim)

        top_sims.sort(reverse=True)
        num_words_to_select = min(len(top_sims), top_num_words)

        #print top_sims[:num_words_to_select]
        sim_sum=sum(top_sims[:num_words_to_select])
        if(num_words_to_select>0):
            avg_sim = sim_sum/num_words_to_select

        return avg_sim

    @staticmethod
    def get_question_vec_to_top_words_avg_sim_wordgroups(words1, words2, model, num_features, index2word_set, top_num_words):
        #function to average all words vectors in a given paragraph
        if(len(words1)==0 or len(words2)==0):
            return 0.00

        nwords = 0 #all words checked
        avg_sim=0.00 #avg similarity to be returned

        top_sims=[]

        words1_in_model = words1
        for word2 in words2:
            if word2 in index2word_set:
                nwords = nwords+1
                sim = model.n_similarity(words1_in_model, [word2])
                top_sims.append(sim)

        top_sims.sort(reverse=True)
        num_words_to_select = min(len(top_sims), top_num_words)

        #print top_sims[:num_words_to_select]
        sim_sum=sum(top_sims[:num_words_to_select])
        if(num_words_to_select>0):
            avg_sim = sim_sum/num_words_to_select

        return avg_sim

    @staticmethod
    def getAvgFeatureVecs(doc_wordlists, model, num_features, index2word_set=set()):
        counter = 0
        reviewFeatureVecs = np.zeros((len(doc_wordlists), num_features), dtype="float32")

        #pass index2word_set as if used more than once outside this function
        if(len(index2word_set)==0):
            index2word_set = set(model.index2word)

        for doc_wordlist in doc_wordlists:
            if((counter%1000) == 0):
                print "Doc %d of %d" %(counter, len(doc_wordlists))
                print doc_wordlist

            reviewFeatureVecs[counter] = AverageVectorsUtilities.makeFeatureVec(doc_wordlist, model,\
                                                                                                    num_features,index2word_set)
            counter += 1

        return reviewFeatureVecs

    @staticmethod
    def get_vectors_from_model_by_id(ids_list, texts_list, model, num_features, index2word_set=set()):
        counter = 0
        reviewFeatureVecs = np.zeros((len(ids_list), num_features), dtype="float32")

        if(len(index2word_set)==0):
            index2word_set = set(model.index2word)

        for i in range(0,len(ids_list)):
            if((counter%1000) == 0):
                print "Doc %d of %d" %(counter, len(ids_list))

            doc_id = ids_list[i]
            if doc_id in index2word_set:
                reviewFeatureVecs[counter] = model[doc_id]
            else:
                reviewFeatureVecs[counter] = AverageVectorsUtilities.makeFeatureVec(texts_list[i], model,\
                                                                                                    num_features,index2word_set)
            counter += 1

        return reviewFeatureVecs

    @staticmethod
    def get_sim_top_most_similar_items(vector1, list_of_vectors, list_of_vector_keys, list_of_vector_data, top_n_sim, category='', all_data=None):
        similar_vectors=[]
        for i in range(0, len(list_of_vectors)):
            if category!='' and all_data[i]!=category:
                continue
            sim = 1 - spatial.distance.cosine(vector1,list_of_vectors[i])
            similar_vectors.append(((list_of_vector_keys[i], list_of_vector_data[i]), sim))

        similar_vectors.sort(key=lambda x:x[1], reverse=True)
        return similar_vectors[:top_n_sim]



#SAMPLE TEST USAGE
#python Word2Vec_AverageVectorsUtilities.py E:\semeval2016-task3-caq\qatarliving\qatarliving_qc_size100_win10_mincnt5_with_sent_repl_iter1.word2vec.bin
if __name__=='__main__':
    import logging
    from gensim.models.word2vec import Word2Vec
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #read publications file
    if len(sys.argv) > 1:
        word2vec_file_to_load = sys.argv[1]
        print('Word2vec file:\n\t%s' % word2vec_file_to_load)
    else:
        print('Error: missing input file parameter')
        quit()

    model = Word2Vec.load(word2vec_file_to_load)
    index2word = set(model.index2word)

    word2vec_num_features = len(model.syn0[0])
    print "Feature vectors length:%s"%word2vec_num_features
    print "Model syn0 len=%d"%(len(model.syn0))

    question_body = u'is there any place i can find scented massage oils in qatar?'
    answers = [u'Yes. It is right behind Kahrama in the National area.',\
                    u'whats the name of the shop?',\
                    u'It s called Naseem Al-Nadir. Right next to the Smartlink shop. You ll find the chinese salesgirls at affordable prices there.',\
                    u'dont want girls;want oil',\
                    u'Try Both ;) I am just trying to be helpful. On a serious note - Please go there. you ll find what you are looking for.',\
                    u'you mean oil and filter both',\
                    u'Yes Lawa...you couldn t be more right LOL',\
                    u'What they offer?',\
                    u'FU did u try with that salesgirl ?',\
                    u'Swine - No I don t try with salesgirls. My taste is classy ']

    print question_body
    print "Average best similarity - q words to best c word"
    for x in answers:
        sim = AverageVectorsUtilities.get_feature_vec_avg_aligned_sim(question_body.lower().split(), x.lower().split(), model, word2vec_num_features, index2word)
        print "%s\t%s"%(sim, x)

    print "Question to best N words"
    print "sim\tsim_top1\tsim_top2\tsim_top3\tsim_top5"
    for x in answers:
        words1=question_body.lower().split()
        words2=x.lower().split()
        sim = AverageVectorsUtilities.get_feature_vec_avg_aligned_sim(words1, words2, model, word2vec_num_features, index2word)
        sim_top1 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, model, word2vec_num_features, index2word, 1)
        sim_top2 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, model, word2vec_num_features, index2word, 2)
        sim_top3 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, model, word2vec_num_features, index2word, 3)
        sim_top5 = AverageVectorsUtilities.get_question_vec_to_top_words_avg_sim(words1, words2, model, word2vec_num_features, index2word, 5)
        print "%s\t%s\t%s\t%s\t%s\t%s"%(sim, sim_top1, sim_top2, sim_top3, sim_top5, x)


