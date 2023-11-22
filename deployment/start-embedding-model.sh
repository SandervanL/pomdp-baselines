#!/bin/bash

# Define the list of seeds until 72
seeds=(42)

# Define the list of gammas
task_files=("sentences_word2vec" "sentences_simcse" "sentences_sbert" "sentences_word2vec_pos")

working_dir=$(pwd)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for task_file in "${task_files[@]}"; do
    sbatch -J "model-$task_file" sbatch-embedding-model.sh $working_dir $seed $task_file 48
  done
done

sbatch -J "model-sentences_infersent-42" --mem-per-cpu 4G --cpus-per-task 24 sbatch-embedding-model.sh $working_dir 42 "sentences_infersent" 24
sbatch -J "model-sentences_infersent-66" --mem-per-cpu 4G --cpus-per-task 24 sbatch-embedding-model.sh $working_dir 66 "sentences_infersent" 24
