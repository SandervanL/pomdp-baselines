#!/bin/bash

#SBATCH --job-name="run-blocks"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1	          # this is equivalent to number of NODES
##SBATCH --gpus-per-task=4  # comment out if not needing GPUs
#SBATCH --cpus-per-task=1     # usually recommended in powers of 2 or divisible by 2
#SBATCH --partition=compute	  # possible partitions are: <compute gpu memory> The standard is compute, memory is for high memory requirements.
##SBATCH --mem-per-gpu=20GB	  # only if requesting gpus, also mutually exclusive with --mem and --mem-per-cpu
#SBATCH --mem=8G	          # how much RAM is your job going to require
#SBATCH --account=Education-EEMCS-MSc-CS	# can comment out and than uses the default innovation account that all scientific stuff have access to.

# Setup modules
module load 2022r2	# main module, top of the module hierarchy
module load miniconda3	# if you want conda
#module load 2022r2 cuda/11.7 # cudnn/8.0.5.39-11.1	# loads cuda. My jobs that use cudnn do not require me to load cudnn, with the right version of torch

# Set conda env, if you need to:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda env, if you need to:
conda activate rl-lang-11

# I like to print the activated env and its location, to see that it loaded what I expect it to:
echo $CONDA_DEFAULT_ENV
echo $CONDA_PREFIX

# Print the GPU-utilization output, to see that we get the resources we expect (only relevant when GPUs are used):
#/usr/bin/nvidia-smi

# to view GPU resource utilization at the end of the job, setup:
#previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Call your script
srun python /home/sajvanleeuwen/code/src/main.py \
  --algorithm dqn drqn \
  --exploration epsilon-greedy rnd \
  --env block0-maze-v0 block1-maze-v0 block2-maze-v0 block3-maze-v0 block4-maze-v0 block6-maze-v0 block8-maze-v0 block12-maze-v0 \
  --executions 1 \
  --no-plot \
  --log-frequency 100 \
  --no-print-when-plot \
  --max-episode-length 400 \
  --max-episodes 1000 \
  --learning-rate 0.0002 0.0005 0.001 0.002 0.005 0.01 \
  --epsilon-anneal-time 2500 5000 6000 7500 10000 13000 \
  --save-dir /home/sajvanleeuwen/logs/blocks_anneal_test

# the other part of view GPU resource utilization:
#/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate