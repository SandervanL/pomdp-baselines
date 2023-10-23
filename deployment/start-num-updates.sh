#!/bin/bash

# Define the list of seeds
#seeds=(42 43 44 45 46 47 48 49 50 51)

# Define the list of gammas
num_updates_per_iters=(0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2)

working_dir=$(pwd)

# Echo the list of seeds
#for seed in "${seeds[@]}"; do
  for updates in "${num_updates_per_iters[@]}"; do
    sbatch -J "$updates-iters" sbatch-num-updates.sh $working_dir 42 $updates
  done
#done
