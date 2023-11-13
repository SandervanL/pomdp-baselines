#!/bin/bash

# Define the list of seeds
seeds=(42)

# Define the list of task files
task_files=("object_type_simcse" "object_type_word2vec" "sentences_simcse" "sentences_word2vec" "words_simcse" "words_word2vec")

working_dir=$(pwd)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for task_file in "${task_files[@]}"; do
    sbatch -J "$task_file-$seed-info" sbatch-embedding-type.sh $working_dir $seed $task_file
  done
done
