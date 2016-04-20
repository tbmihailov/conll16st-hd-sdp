#!/usr/bin/env bash

# set params
input_dataset_train=data/conll16st-en-01-12-16-trial

run_type=svm
run_name=${runtype}_sup_v1

#output dir for parsing results - used for test operations
output_dir=output/${run_name}
mkdir -p ${output_dir}

#model dir where output models are saved after train
model_dir=models/$run_name
mkdir -p ${model_dir}

#resources
word2vec_model=resources/external/w2v_embeddings/qatarliving_qc_size100_win5_mincnt1_skip3_with_sent_repl_iter1.word2vec.bin

word2vec_load_bin=False
#word2vec_load_bin=True - for google pretrained embeddings

#run parser in train mode
mode=train
python sup_parser_v1.py en ${input_dataset_train} ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin}