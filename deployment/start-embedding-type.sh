#!/bin/bash

# Define the list of seeds until 72
seeds=(42)

# Define the list of gammas
task_files=("sentences_word2vec.dill" "words_word2vec.dill" "sentences_simcse.dill" "words_simcse.dill" "object_type_word2vec.dill" "object_type_simcse.dill")

working_dir=$(pwd)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for task_file in "${task_files[@]}"; do
    sbatch -J "$task_file-$seed" sbatch-embedding-type.sh $working_dir $seed $task_file
  done
done
