#!/usr/bin/env bash

echo ----params----
echo input_dataset_train:${input_dataset_train}
echo input_dataset_train_short_name:${input_dataset_train_short_name}

echo input_dataset_test:${input_dataset_test}
echo input_dataset_test_short_name:${input_dataset_test_short_name}

echo run_type:${run_type}
echo run_name:${run_name}
echo output_dir:${output_dir}
echo word2vec_model:${word2vec_model}
echo word2vec_load_bin:${word2vec_load_bin}


#run parser in train mode
echo '=========================================='
echo '==============TRAIN======================='
echo '=========================================='
#mode=train
#dataset_name=${input_dataset_train_short_name}
#echo python sdp/sup_parser_v2_hierarchy_ext.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features} -dataset_name:${dataset_name}
#python sdp/sup_parser_v2_hierarchy_ext.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features} -dataset_name:${dataset_name}

echo '=========================================='
echo '==============TEST========================'
echo '=========================================='
mode=test
dataset_name=${input_dataset_test_short_name}
echo "python sdp/sup_parser_v2_hierarchy_ext.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features} -dataset_name:${dataset_name}"
python sdp/sup_parser_v2_hierarchy_ext_multi_file_eval.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features} -dataset_name:${dataset_name}


# validate output
# python scorer/validator.py en ${output_dir}/output.json

#score
# python scorer/scorer/scorer.py ${input_dataset_test}/relations.json ${output_dir}/output.json

#python scorer/scorer/tira_sup_eval.py ${input_dataset_test} ${output_dir} ${output_dir}