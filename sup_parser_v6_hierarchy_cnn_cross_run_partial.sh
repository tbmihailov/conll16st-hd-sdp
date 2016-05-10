#!/usr/bin/env bash

echo ----params----
echo input_dataset_train:${input_dataset_train}
echo input_dataset_test:${input_dataset_test}
echo run_type:${run_type}
echo run_name:${run_name}
echo output_dir:${output_dir}
echo word2vec_model:${word2vec_model}
echo word2vec_load_bin:${word2vec_load_bin}


#run parser in train mode
echo '=========================================='
echo '==============TRAIN======================='
echo '=========================================='
mode=train
echo python sup_parser_v6_hierarchy_cnn_cross.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features}
python sup_parser_v6_hierarchy_cnn_cross.py en ${input_dataset_train}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features}

# exit

echo '=========================================='
echo '==============TEST========================'
echo '=========================================='
mode=test
echo python sup_parser_v6_hierarchy_cnn_cross.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features}
python sup_parser_v6_hierarchy_cnn_cross.py en ${input_dataset_test}  ${model_dir} ${output_dir} -run_name:${run_name} -cmd:${mode} -word2vec_model:${word2vec_model} -word2vec_load_bin:${word2vec_load_bin} -scale_features:${scale_features}


# validate output
# python validator.py en ${output_dir}/output.json

#score
# python scorer.py ${input_dataset_test}/relations.json ${output_dir}/output.json

python tira_sup_eval.py ${input_dataset_test} ${output_dir} ${output_dir}