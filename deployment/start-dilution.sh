#!/bin/bash

# Define the list of seeds
seeds=(42)

# Define the list of task files
task_files=("object_type_simcse.dill" "object_type_word2vec.dill" "sentences_simcse.dill" "sentences_word2vec.dill" "words_simcse.dill" "words_word2vec.dill")

working_dir=$(pwd)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for task_file in "${task_files[@]}"; do
    sbatch -J "$task_file-$seed-info" sbatch-embedding-type.sh $working_dir $seed $task_file
  done
done
