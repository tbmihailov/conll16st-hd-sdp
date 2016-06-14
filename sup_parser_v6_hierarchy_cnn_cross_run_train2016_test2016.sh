#!/usr/bin/env bash

# set params
#input_dataset_train=data/conll16st-en-01-12-16-trial
#input_dataset_test=data/conll16st-en-01-12-16-trial

input_dataset_train=data/conll16-st-train-en-2016-03-29
input_dataset_test=data/conll16st-en-03-29-16-test

run_type=svm_cnn

run_name=${run_type}_sup_v6_tr16_test16
if [ -n "$1" ]
then
  run_name=$1
fi     # $String is null.


#output dir for parsing results - used for test operations
output_dir=output/${run_name}
mkdir -p ${output_dir}

#model dir where output models are saved after train
model_dir=models/${run_name}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}

scale_features=True

# resources
# word2vec_model=resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin
word2vec_model=resources/closed_track/word2vec_google/GoogleNews-vectors-negative300.bin

# word2vec_load_bin=False
word2vec_load_bin=True # for google pretrained embeddings

log_file=${run_name}_$(date +%y-%m-%d-%H-%M).log
. sup_parser_v6_hierarchy_cnn_cross_run_partial.sh > ${log_file}