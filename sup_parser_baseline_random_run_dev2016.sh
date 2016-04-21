#!/usr/bin/env bash

# set params
# input_dataset_train=data/conll16st-en-01-12-16-trial
# input_dataset_test=data/conll16st-en-01-12-16-trial

# input_dataset_train=data/conll16-st-train-en-2016-03-29
input_dataset_test=data/conll16-st-dev-en-2016-03-29

run_type=random

run_name=${run_type}_sup_v1

# output dir for parsing results - used for test operations
output_dir=output/${run_name}
mkdir -p ${output_dir}

echo '=========================================='
echo '==============RANDOM TEST================='
echo '=========================================='

mode=test
echo python sup_parser_baseline_random.py en ${input_dataset_test}  ${input_dataset_test} ${output_dir}
python sup_parser_baseline_random.py en ${input_dataset_test}  ${input_dataset_test} ${output_dir}

# validate output
python validator.py en ${output_dir}/output.json

# score
python scorer.py ${input_dataset_test}/relations.json ${output_dir}/output.json


