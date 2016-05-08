#!/usr/bin/env bash

# sup_parser_v2_hierarchy_run_tira_test.sh $inputDataset svm_base_sup_v2 $outputDir
# set params
#input_dataset_train=data/conll16st-en-01-12-16-trial
#input_dataset_test=data/conll16st-en-01-12-16-trial

input_dataset_test=$1

run_type=svm_base

run_name=svm_base_sup_v2
if [ -n "$2" ]
then
  run_name=$2
fi     # $String is null.


#output dir for parsing results - used for test operations
output_dir=$3
# mkdir -p ${output_dir}

#model dir where output models are saved after train
model_dir=models/${run_name}
# rm -rf -- ${model_dir}
# mkdir -p ${model_dir}

scale_features=True

# resources
# word2vec_model=resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin
word2vec_model=resources/closed_track/word2vec_google/GoogleNews-vectors-negative300.bin

# word2vec_load_bin=False
word2vec_load_bin=True # for google pretrained embeddings

# log_file=${run_name}_$(date +%y-%m-%d-%H-%M).log

echo ----params----
# echo input_dataset_train:${input_dataset_train}
echo input_dataset_test:${input_dataset_test}
# echo run_type:${run_type}
echo run_name:${run_name}
echo model_dir:${model_dir}
echo output_dir:${output_dir}
echo word2vec_model:${word2vec_model}
echo word2vec_load_bin:${word2vec_load_bin}


#run parser in train mode
#echo '=========================================='
#echo '==============TRAIN======================='
#echo '=========================================='
#mode=train
#echo python sup_parser_v2_hierarchy.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features}
#python sup_parser_v2_hierarchy.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features}

echo '=========================================='
echo '==============TEST========================'
echo '=========================================='
mode=test
echo python sup_parser_v2_hierarchy.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features}
python sup_parser_v2_hierarchy.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features}


# validate output
# python validator.py en ${output_dir}/output.json

#score
# python scorer.py ${input_dataset_test}/relations.json ${output_dir}/output.json

# python tira_sup_eval.py ${input_dataset_test} ${output_dir} ${output_dir}