#!/bin/bash

# Define the list of seeds
seeds=(42 43 44 45 46 47 48 49 50 51)

# Define the list of gammas
task_selections=("random-word")

working_dir=$(pwd)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for task_selection in "${task_selections[@]}"; do
    sbatch -J "$task_selection-$seed" sbatch-sentence.sh $working_dir $seed $task_selection
    sleep 1
  done
done
