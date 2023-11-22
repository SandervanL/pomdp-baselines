#!/bin/bash

# Define the list of seeds until 72
seeds=(42)

# Define the list of gammas
activations=("grad-leaky" "grad-swish" "grad-elu" "grad-selu" "grad-gelu" "grad-relu6")

working_dir=$(pwd)

# Echo the list of seeds
for seed in "${seeds[@]}"; do
  for activation in "${activation[@]}"; do
    sbatch -J $activation sbatch-activation.sh $working_dir $seed $activation
  done
done
