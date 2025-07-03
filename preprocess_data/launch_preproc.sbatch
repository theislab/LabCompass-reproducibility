#!/bin/bash

#SBATCH --job-name=launch_preprocessing
#SBATCH --output=run_preprocessing.out
#SBATCH --error=run_preprocessing.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=1T
#SBATCH --cpus-per-task=32
#SBATCH --nice=10000
#SBATCH --time=3-00:00:00


## set environment name
ENV_NAME="rapids_singlecell";
echo "Using environment  $ENV_NAME";
conda run -n $ENV_NAME python ./run_preprocessing.py;
