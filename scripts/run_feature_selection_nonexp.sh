#!/usr/bin/env bash


model_dir=/home/mihaylov/Programming/conll16st-hd-sdp/models/svm_base_sup_v2_hier_ext_devdev
prefix=${model_dir}/svm_base_sup_v2_hier_ext_devdev
input_libsvm_file=${prefix}_scalerange__NONEXP_LEVEL1_scale_dev2016.libsvm
#features_filter_file=features_filter_nonexp_all.features_filter
features_filter_file=
features_dict_file=${prefix}_model__NONEXP_LEVEL1.features
class_mappings_file=${prefix}_model_.classmapping
python LibSvm_Data_FeatureSelection.py -input_libsvm_file:${input_libsvm_file} -features_dict_file:${features_dict_file} -features_filter_file:${features_filter_file} -class_mappings_file:${class_mappings_file}