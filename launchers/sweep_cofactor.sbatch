#!/bin/bash

#SBATCH --job-name=sweep_cof
#SBATCH --output=logs/sweep_cof.out
#SBATCH --error=logs/sweep_cof.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --mem=256G
#SBATCH --cpus-per-task=6
#SBATCH --nice=0
#SBATCH --time=3-00:00:00

## set environment name
ENV_NAME="sc_exp_design";
echo "Using environment  $ENV_NAME";

## hydra error logging
export HYDRA_FULL_ERROR=1;

## set working directory
WDIR=./scripts;
cd $WDIR;
echo "Current Working Directory $WDIR";

## define configuration variables
CONFIG_NAME="train_flow"
GOETTGENS_CONFIG_PATH=../config;

## Spectral Flow Cytometry unconditional
echo "Running experiment on Spectral Flow Cytometry data in the unconditional case";
conda run -n $ENV_NAME python ./train_flow.py --multirun --config-path $GOETTGENS_CONFIG_PATH --config-name $CONFIG_NAME \
        +hparams_search=train_flow_unconditional_sweep_cofactor +launcher=slurm \
         callbacks.wandb_project_name="SFC_Goettgens";
