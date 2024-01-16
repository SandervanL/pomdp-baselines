#!/bin/bash

# Define the list of seeds until 89
seeds=(42)

# Define the list of gammas
task_selections=("random-word" "random-within-word")

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for task_selection in "${task_selections[@]}"; do
    sbatch -J "$task_selection" sbatch-sentence.sh $(pwd) $seed $task_selection
  done
done
