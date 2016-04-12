import sys
from pandas import json
import codecs
from Common_Utilities import CommonUtilities


class Conll2016DataUtilities(object):
    @staticmethod
    # Read relations data from pdtb file - relations.json
    # http://nbviewer.jupyter.org/github/attapol/conll16st/blob/master/tutorial/tutorial.ipynb#relations.json-:-Gold-standard-discourse-relation-annotation
    def read_relations_from_pdtb_file(file_name):
        relations = []
        with codecs.open(file_name, mode='r', encoding='utf-8') as pdtb_file:
            relations = [json.loads(x) for x in pdtb_file]

        return relations

    @staticmethod
    # Read data from input file parse.json
    # http://nbviewer.jupyter.org/github/attapol/conll16st/blob/master/tutorial/tutorial.ipynb#parses.json-:-Input-for-the-main-task-and-the-supplementary-task
    def read_input_data_from_parse_file(file_name):
        with codecs.open(file_name, mode='r', encoding='utf-8') as parse_file:
            json_str = parse_file.read().strip()
            print json_str
            en_parse_dict = json.loads(json_str)

        return en_parse_dict


# SAMPLE USAGE
# python conll16_datautilities.py -rel_file:tutorial\conll16st-en-01-12-16-trial\relations.json -parse_file:tutorial\conll16st-en-01-12-16-trial\parses.json -sup_task_rel_file:tutorial\conll16st-en-01-12-16-trial\relations-no-senses.json
if __name__ == '__main__':
    relations_file = CommonUtilities.get_param_value('rel_file', sys.argv, "")
    if relations_file == "":
        raise "please, specify -rel_file:tutorial\conll16st-en-01-12-16-trial\relations.json"

    parse_file = CommonUtilities.get_param_value('parse_file', sys.argv, "")
    if parse_file == "":
        raise "please, specify -parse_file:tutorial\conll16st-en-01-12-16-trial\parses.json"

    sup_task_rel_file = CommonUtilities.get_param_value('sup_task_rel_file', sys.argv, "")
    if sup_task_rel_file == "":
        raise "please, specify -sup_task_rel_file:tutorial\conll16st-en-01-12-16-trial\relations-no-senses.json"

    relations = Conll2016DataUtilities.read_relations_from_pdtb_file(relations_file)
    print "%s relations found!" % len(relations)
    print "example relation [0]:"
    # print relations[0]

    en_parse_dict = Conll2016DataUtilities.read_input_data_from_parse_file(parse_file)

    # example relation
    en_example_relation = relations[10]
    en_doc_id = en_example_relation["DocID"]

    # en parse tree
    en_parse_tree = en_parse_dict[en_doc_id]["sentences"][15]["parsetree"]
    print "en parse tree:"
    print en_parse_tree

    # en dependencies
    en_dependencies = en_parse_dict[en_doc_id]['sentences'][15]['dependencies']
    print "en dependencies:"
    print en_dependencies

    # en single word info
    en_single_word_info = en_parse_dict[en_doc_id]['sentences'][15]['words'][0]
    print "en single word info:"
    print en_single_word_info

