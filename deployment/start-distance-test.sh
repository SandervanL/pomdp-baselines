#!/bin/bash

# Define the list of seeds
seeds=(42 43 44 45 46 47 48 49 50 51)

# Define the list of gammas
gammas=(0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99)

working_dir=$(pwd)

# Echo the list of seeds
echo "Seeds:"
for seed in "${seeds[@]}"; do
  for gamma in "${gammas[@]}"; do
    sbatch -J "$gamma-$seed-embedding" sbatch-distance-test.sh $working_dir $seed $gamma
    sleep 1
  done
done
