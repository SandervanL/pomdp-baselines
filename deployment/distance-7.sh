#!/bin/bash

#SBATCH --job-name="distance-7-no-penalty"
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
echo "Conda Default Env: ${CONDA_DEFAULT_ENV}"
echo "Conda Prefix: ${CONDA_PREFIX}"

# Print the GPU-utilization output, to see that we get the resources we expect (only relevant when GPUs are used):
#/usr/bin/nvidia-smi

# to view GPU resource utilization at the end of the job, setup:
#previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
source_file_path=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}' | head -n 1)
echo "Source_file_path: ${source_file_path}"
dir_path=$(dirname ${source_file_path})
echo "dir path: ${dir_path}"
project_path=$dir_path/..
echo "project path: ${project_path}"

# Call your script
srun python $project_path/main.py --cfg $project_path/configs/pomdp/maze/blocked-7.yml

# the other part of view GPU resource utilization:
#/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate