#!/bin/bash

#SBATCH --job-name=launch_preprocessing
#SBATCH --output=run_preprocessing.out
#SBATCH --error=run_preprocessing.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --mem=500G
#SBATCH --cpus-per-task=6
#SBATCH --nice=10000
#SBATCH --time=2-00:00:00


## set environment name
ENV_NAME="rapids_singlecell";
echo "Using environment  $ENV_NAME";
conda run -n $ENV_NAME python ./run_preprocessing.py