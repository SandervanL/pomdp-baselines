#!/bin/bash

# Define the list of seeds
seeds=(42 43 44 45 46 47 48 49 50 51)

# Define the list of gammas
init_levels=(0 1 2 3)

working_dir=$(pwd)

# Echo the list of seeds
echo "Seeds:"
for seed in "${seeds[@]}"; do
  for obs_init in "${init_levels[@]}"; do
    for rnn_init in "${init_levels[@]}"; do
      sbatch -J "$obs_init-$rnn_init-$seed-embedding" embedding-test.sh $working_dir $obs_init $rnn_init $seed
      sleep 1
    done
  done
done
