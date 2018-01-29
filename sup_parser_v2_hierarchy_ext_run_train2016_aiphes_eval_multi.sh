#!/usr/bin/env bash

# set params
#input_dataset_train=data/conll16st-en-01-12-16-trial
#input_dataset_test=data/conll16st-en-01-12-16-trial

input_dataset_train=data/conll16-st-train-en-2016-03-29
input_dataset_train_short_name=tr2016

input_dataset_test=data/conll16-st-dev-en-2016-03-29
input_dataset_test_short_name=dev2016

run_type=svm_base

run_name=${run_type}_sup_v2_hier_ext_tr16dev-aiphes
if [ -n "$1" ]
then
  run_name=$1
fi     # $String is null.


#model dir where output models are saved after train
model_dir=models/${run_name}
#rm -rf -- ${model_dir}
#mkdir -p ${model_dir}

scale_features=True

# resources
# word2vec_model=resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin
word2vec_model=resources/closed_track/word2vec_google/GoogleNews-vectors-negative300.bin

# word2vec_load_bin=False
word2vec_load_bin=True # for google pretrained embeddings



output_base_dir=/home/mitarb/mihaylov/research/data/aiphes-summarization-collab/test_conll16st
output_dirs=(
${output_base_dir}/DUC2003
${output_base_dir}/DUC2004
${output_base_dir}/TAC2008A
${output_base_dir}/TAC2009A
${output_base_dir}/hMDS_A
${output_base_dir}/hMDS_M
${output_base_dir}/hMDS_V
)

input_base_dir=/home/mitarb/mihaylov/research/data/aiphes-summarization-collab/test_conll16st
input_dirs=(
${input_base_dir}/DUC2003
${input_base_dir}/DUC2004
${input_base_dir}/TAC2008A
${input_base_dir}/TAC2009A
${input_base_dir}/hMDS_A
${input_base_dir}/hMDS_M
${input_base_dir}/hMDS_V
)

dataset_names_short=(
DUC2003
DUC2004
TAC2008A
TAC2009A
hMDS_A
hMDS_M
hMDS_V
)

coreNlpPath="/home/mitarb/mihaylov/research/libs/corenlp/stanford-corenlp-full-2015-12-09/*;"

files_cnt=7
for ((i=0;i<files_cnt;i++)); do
    echo "---------------"
    input_dir=${input_dirs[${i}]}
    output_dir=${output_dirs[${i}]}

    echo "input_dir: ${input_dir}"
    echo "input_dir: ${input_dir}"

    input_dataset_test=data/conll16-st-dev-en-2016-03-29
    input_dataset_test_short_name=dev2016

    script_name=raw_text_to_json_run.py
    run_name=raw_text_to_json_run_${i}
    log_file=${input_dir}_dr_$(date +%y-%m-%d-%H-%M-%S).log
    . ~/tools/notify/script_started.sh

    log_file=${run_name}_$(date +%y-%m-%d-%H-%M).log
    . sup_parser_v2_hierarchy_ext_run_partial_aiphes.sh > ${log_file}

    . ~/tools/notify/script_stopped.sh

    echo "---------------"
done