#!/bin/bash

# Define the list of seeds until 89
seeds=(42)

# Define the list of gammas
uncertainties=("0.01" "0.2" "0.5" "1" "2" "5" "10" "20" "50" "100")

working_dir=$(pwd)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for uncertainty in "${uncertainties[@]}"; do
    sbatch -J "$uncertainty-$seed-uncertainty" sbatch-sentence.sh $working_dir $seed $uncertainty
  done
done
