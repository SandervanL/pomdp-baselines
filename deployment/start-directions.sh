#!/bin/bash

# Define the list of seeds until 72
seeds=(42)

# Define the list of gammas
task_files=("left_longhook.dill" "leftright_longhook.dill" "left_longstraight.dill" "leftright_longstraight.dill" "left_allstraight.dill" "leftright_allstraight.dill")

working_dir=$(pwd)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for task_file in "${task_files[@]}"; do
    sbatch -J "$task_file-$seed" sbatch-directions.sh $working_dir $seed $task_file
  done
done
