#!/usr/bin/env bash

echo ----params----
echo input_dataset_train:${input_dataset_train}
echo input_dataset_test:${input_dataset_test}
echo run_type:${run_type}
echo run_name:${run_name}
echo output_dir:${output_dir}
echo word2vec_model:${word2vec_model}
echo word2vec_load_bin:${word2vec_load_bin}

echo deps_model:${deps_model}
echo brownclusters_file:${brownclusters_file}


#run parser in train mode
echo '=========================================='
echo '==============TRAIN======================='
echo '=========================================='
mode=train
echo python sdp/sup_parser_v2_hierarchy_optimized.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features} -deps_model:${deps_model} -brownclusters_file:${brownclusters_file}
python sdp/sup_parser_v2_hierarchy_optimized.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features} -deps_model:${deps_model} -brownclusters_file:${brownclusters_file}

echo '=========================================='
echo '==============TEST========================'
echo '=========================================='
mode=test
echo python sdp/sup_parser_v2_hierarchy_optimized.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features} -deps_model:${deps_model} -brownclusters_file:${brownclusters_file}
python sdp/sup_parser_v2_hierarchy_optimized.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features} -deps_model:${deps_model} -brownclusters_file:${brownclusters_file}


# validate output
# python validator.py en ${output_dir}/output.json

#score
# python scorer.py ${input_dataset_test}/relations.json ${output_dir}/output.json

python scorer/tira_sup_eval.py ${input_dataset_test} ${output_dir} ${output_dir}