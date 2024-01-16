#!/bin/bash

# Define the list of seeds until 72
seeds=(42)

# Define the list of gammas
task_files=(
 "all_baseline"
 "all_negation_baseline"
 "all_negation_new_sentences"
 "all_negation_perfect"
 "all_new_sentences"
 "all_perfect"
 "left_baseline"
 "left_lr_orig_sentences"
 "left_new_sentences"
 "left_orig_sentences"
 "left_perfect"
 "left_right_baseline"
 "left_right_new_sentences"
 "left_right_orig_sentences"
 "left_right_perfect"
 "right_baseline"
 "right_new_sentences"
 "right_orig_sentences"
 "right_perfect"
)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for task_file in "${task_files[@]}"; do
    sbatch -J "directions-$task_file" sbatch-directions-paper.sh $(pwd) $seed $task_file
  done
done
