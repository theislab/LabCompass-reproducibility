#!/bin/bash

#SBATCH --job-name=plotting
#SBATCH --output=logs/run_plots.out
#SBATCH --error=logs/run_plots.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=1T
#SBATCH --cpus-per-task=6
#SBATCH --nice=10000
#SBATCH --time=3-00:00:00

## set working directory
WDIR=./plotting;
cd $WDIR;
echo "Current Working Directory $WDIR";

## set environment name
ENV_NAME="rapids_singlecell";
echo "Using environment  $ENV_NAME";
conda run -n $ENV_NAME python ./run_plotting.py;
