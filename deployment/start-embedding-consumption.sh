#!/bin/bash

# Define the list of seeds
seeds=(42)

# Define the list of gammas
obs_init_levels=(2 3)
rnn_init_levels=(0 1 2 3)

working_dir=$(pwd)

for seed in "${seeds[@]}"; do
  for obs_init in "${obs_init_levels[@]}"; do
    for rnn_init in "${rnn_init_levels[@]}"; do
      sbatch -J "$obs_init-$rnn_init-embedding" sbatch-embedding-consumption.sh $working_dir $obs_init $rnn_init $seed
    done
  done
done

#for seed in "${seeds[@]}"; do
#  sbatch -J "0-0-embedding" sbatch-embedding-consumption.sh $working_dir 0 0 $seed
#  sbatch -J "0-1-embedding" sbatch-embedding-consumption.sh $working_dir 0 1 $seed
#  sbatch -J "0-3-embedding" sbatch-embedding-consumption.sh $working_dir 0 3 $seed
#  sbatch -J "1-0-embedding" sbatch-embedding-consumption.sh $working_dir 1 0 $seed
#  sbatch -J "2-0-embedding" sbatch-embedding-consumption.sh $working_dir 2 0 $seed
#done

