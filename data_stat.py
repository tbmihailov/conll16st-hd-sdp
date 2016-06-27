import sys
import json

# Example from http://www.cs.brandeis.edu//~clp/conll15st/blog/data_statistics.html
# python data_stat.py data/conll15st-en-03-29-16-blind-test > data/stat_conll15st-en-03-29-16-blind-test.txt
# python data_stat.py data/conll16-st-dev-en-2016-03-29 > data/stat_conll16-st-dev-en-2016-03-29.txt
# python data_stat.py data/conll16st-en-03-29-16-test > data/stat_conll16st-en-03-29-16-test.txt
# python data_stat.py data/conll16-st-dev-en-2016-03-29 > data/stat_conll16-st-dev-en-2016-03-29.txt
# python data_stat.py data/conll16-st-train-en-2016-03-29 > data/stat_conll16-st-train-en-2016-03-29.txt
input_dataset = sys.argv[1]

relation_file = '%s/relations.json' % input_dataset  # with senses to train
relation_dicts = [json.loads(x) for x in open(relation_file)]

print input_dataset

print '------------------------------------'
sense_dist = {}
type_dist = {}
sense_type_dist = {}
for relation in relation_dicts:
    if relation['Type']=='Explicit':
        type = 'Explicit'
    else:
        type = 'Non-Explicit'

    for s in relation['Sense']:
        if type in type_dist:
            type_dist[type] += 1
        else:
            type_dist[type] = 1

        if 'All' in type_dist:
            type_dist['All'] += 1
        else:
            type_dist['All'] = 1

        if s in sense_dist:
            sense_dist[s] += 1
        else:
            sense_dist[s] = 1

        if type+'_'+s in sense_type_dist:
            sense_type_dist[type+'_'+s] += 1
        else:
            sense_type_dist[type+'_'+s] = 1

print '------------------------------------'
print '---------By Sense---------------'
print '------------------------------------'
for k, v in sorted([(x, y) for (x, y) in sense_dist.iteritems()], key=lambda t : t[1]):
    print '\t %s\t%s\t%s \t' % (k, v, round(float(v)/len(relation_dicts) * 100, 2))

print '------------------------------------'
print '---------By Type---------------'
print '------------------------------------'
for k, v in sorted([(x, y) for (x, y) in type_dist.iteritems()], key=lambda t : t[1]):
    print '\t %s\t%s\t%s \t' % (k, v, round(float(v)/len(relation_dicts) * 100, 2))

print '------------------------------------'
print '---------By Sense---------------'
print '------------------------------------'
for k, v in sorted([(x, y) for (x, y) in sense_type_dist.iteritems()], key=lambda t : t[0]):
    print '\t %s\t%s\t%s \t' % (k, v, round(float(v)/len(relation_dicts) * 100, 2))