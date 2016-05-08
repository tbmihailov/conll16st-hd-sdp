import sys

class CommonUtilities(object):
    @staticmethod
    def write_dictionary_to_file(dictionary, file_name, sort_by_key=False):
        with open(file_name, "wb") as f:
            if sort_by_key:
                for key in sorted(dictionary):
                    f.write("%s\t%s\n" %(key, dictionary[key]))
            else:
                for key, val in dictionary.iteritems():
                    f.write("%s\t%s\n" %(key, val))

    @staticmethod
    def load_dictionary_from_file(file_name):
        dictionary = {}
        with open(file_name, "rb") as f:
            for line in f:
                items = line.split("\t")
                dictionary[items[0]]= items[1].replace("\n","")

        return dictionary

    @staticmethod
    def load_labeledsentences_as_lists_from_file(file_name):
        labels = []
        sentences = []
        with open(file_name, "rb") as f:
            for line in f:
                items = line.split("\t")
                labels.append(items[0])
                sentences.append(items[1].replace("\n",""))

        return labels, sentences

    @staticmethod
    def get_param_value(param_name, argv, default=""):
        param_search = "-%s:"%param_name
        val = default

        try:
            match_params = [x for x in argv if x.startswith(param_search)]
            if(len(match_params)==0):
                return val

            param_with_val=match_params[0]
            if(not param_with_val is None and param_with_val!=""):
                val = param_with_val.split(":",1)[1]

        except:
             print "Unexpected error:", sys.exc_info()[0]
        return val

    @staticmethod
    def get_param_value_bool(param_name, argv, default=False):
        param_search = "-%s:"%param_name
        val = default

        try:
            match_params = [x for x in argv if x.startswith(param_search)]
            if(len(match_params)==0):
                return val

            param_with_val=match_params[0]
            if(not param_with_val is None and param_with_val!=""):
                val_str = param_with_val.split(":",1)[1]
                val=True if val_str=='True' else False
        except:
             print "Unexpected error:", sys.exc_info()[0]

        return val

    @staticmethod
    def get_param_value_int(param_name, argv, default=0):
        param_search = "-%s:"%param_name
        val = default

        try:
            match_params = [x for x in argv if x.startswith(param_search)]
            if(len(match_params)==0):
                return val

            param_with_val=match_params[0]
            if(not param_with_val is None and param_with_val!=""):
                val_str = param_with_val.split(":",1)[1]
                val=int(val_str)
        except:
             print "Unexpected error:", sys.exc_info()[0]

        return val

    @staticmethod
    def get_param_value_float(param_name, argv, default=0):
        param_search = "-%s:"%param_name
        val = default

        try:
            match_params = [x for x in argv if x.startswith(param_search)]
            if(len(match_params)==0):
                return val

            param_with_val=match_params[0]
            if(not param_with_val is None and param_with_val!=""):
                val_str = param_with_val.split(":",1)[1]
                val=float(val_str)
        except:
             print "Unexpected error:", sys.exc_info()[0]

        return val

    @staticmethod
    def increment_feat_val(feats_dict, feat_key, increment_val=1):
        if(feat_key in feats_dict):
            feats_dict[feat_key]+=increment_val
        else:
            feats_dict[feat_key]=increment_val

    @staticmethod
    def append_features_with_vectors(feats, vector, feat_prefix):
        for i in range(0, len(vector)):
            CommonUtilities.increment_feat_val(feats, '%s%s' % (feat_prefix, str(i).zfill(4)), vector[i])


if __name__ == '__main__':
    num_features = 1000 #300   # Word vector dimensionality
    num_features = CommonUtilities.get_param_value_int("num_features", sys.argv, num_features)
    print "num_features:%s"%num_features

    val = CommonUtilities.get_param_value("someparam", sys.argv)
    print "Some param value:%s"%val

    dict_file_name = "qatarliving\\qatarliving_size400_win10_mincnt10_nostopwords.word2vec.bin_c_6315__word_cluster_map.txt"

    print "Loading dict from file %s" % (dict_file_name)
    loaded_dict = CommonUtilities.load_dictionary_from_file(dict_file_name)


