#!/bin/bash

# Define the list of seeds
seeds=(42)

# Define the list of gammas
init_levels=(0 1 2 3)

working_dir=$(pwd)

for seed in "${seeds[@]}"; do
  for obs_init in "${init_levels[@]}"; do
    for rnn_init in "${init_levels[@]}"; do
      # If (obs_init is 0 and rnn_init is in [0, 1, 3]), or (rnn_init is 0 and obs_init in [1, 2]), continue
      if [[ ($obs_init -eq 0 && ($rnn_init -eq 0 || $rnn_init -eq 1 || $rnn_init -eq 3)) || ($rnn_init -eq 0 && ($obs_init -eq 1 || $obs_init -eq 2)) ]]; then
        continue
      fi
      sbatch -J "$obs_init-$rnn_init-$seed-embedding" sbatch-embedding-consumption.sh $working_dir $obs_init $rnn_init $seed
    done
  done
done

#for seed in "${seeds[@]}"; do
#  sbatch -J "0-0-$seed-embedding" sbatch-embedding-consumption.sh $working_dir 0 0 $seed
#  sbatch -J "0-1-$seed-embedding" sbatch-embedding-consumption.sh $working_dir 0 1 $seed
#  sbatch -J "0-3-$seed-embedding" sbatch-embedding-consumption.sh $working_dir 0 3 $seed
#  sbatch -J "1-0-$seed-embedding" sbatch-embedding-consumption.sh $working_dir 1 0 $seed
#  sbatch -J "2-0-$seed-embedding" sbatch-embedding-consumption.sh $working_dir 2 0 $seed
#done

