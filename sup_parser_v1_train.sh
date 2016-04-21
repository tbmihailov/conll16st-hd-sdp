#!/usr/bin/env bash

# set params
# input_dataset_train=data/conll16st-en-01-12-16-trial
# input_dataset_test=data/conll16st-en-01-12-16-trial

input_dataset_train=data/conll16-st-train-en-2016-03-29
input_dataset_test=data/conll16-st-dev-en-2016-03-29

run_type=svm_base
run_name=${run_type}_sup_v1

#output dir for parsing results - used for test operations
output_dir=output/${run_name}
mkdir -p ${output_dir}

#model dir where output models are saved after train
model_dir=models/${run_name}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}


#resources
word2vec_model=resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin

word2vec_load_bin=False
#word2vec_load_bin=True - for google pretrained embeddings

#run parser in train mode
echo '=========================================='
echo '==============TRAIN======================='
echo '=========================================='
mode=train
echo python sup_parser_v1.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin}
python sup_parser_v1.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin}

echo '=========================================='
echo '==============TEST========================'
echo '=========================================='
mode=test
echo python sup_parser_v1.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin}
# python sup_parser_v1.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin}


# validate output
# python validator.py en ${output_dir}/output.json

#score
# python scorer.py ${input_dataset_test}/relations.json ${output_dir}/output.json