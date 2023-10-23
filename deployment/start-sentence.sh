#!/bin/bash

# Define the list of seeds until 72
seeds=(42 58 74)

# Define the list of gammas
task_selections=("random" "random-word" "random-within-word")

working_dir=$(pwd)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for task_selection in "${task_selections[@]}"; do
    sbatch -J "$task_selection-$seed" sbatch-sentence.sh $working_dir $seed $task_selection
  done
done
