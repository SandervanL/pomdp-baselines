#!/bin/bash

# Define the list of seeds
seeds=(42 43 44 45 46 47 48 49 50 51)

# Define the list of gammas
task_files=("sentences_word2vec.dill" "words_word2vec.dill" "sentences_simcse.dill" "words_simcse.dill")

working_dir=$(pwd)

# Echo the list of seeds
echo "Seeds:"
for seed in "${seeds[@]}"; do
  for task_file in "${task_files[@]}"; do
    sbatch -J "$task_file-$seed" sbatch-embedding-type.sh $working_dir $seed $task_file
    sleep 1
  done
done
