#!/bin/bash

#SBATCH --job-name=sweep_flow
#SBATCH --output=logs/sweep_flow.out
#SBATCH --error=logs/sweep_flow.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=6
#SBATCH --nice=10000
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
        +hparams_search=train_flow_unconditional +launcher=slurm \
         callbacks.wandb_project_name="SFC_Goettgens" dispatcher.h5ad_data_kwargs.cofactor=3000;

## Spectral Flow Cytometry one hot protocols
# echo "Running experiment on Spectral Flow Cytometry data with one hot encoded protocols";
# conda run -n $ENV_NAME python ./train_flow.py --multirun --config-path $GOETTGENS_CONFIG_PATH --config-name $CONFIG_NAME \
#        +hparams_search=train_flow_one_hot_protocols +launcher=slurm dispatcher.h5ad_data_kwargs.cofactor=3000\
#          callbacks.wandb_project_name="SFC_Goettgens" dispatcher=one_hot_protocols data=one_hot_protocols vf=one_hot_protocols;

## Spectral Flow Cytometry one hot protocol axis
# echo "Running experiment on Spectral Flow Cytometry data with one hot encoded protocol axis";
# conda run -n $ENV_NAME python ./train_flow.py --multirun --config-path $GOETTGENS_CONFIG_PATH --config-name $CONFIG_NAME \
#        +hparams_search=train_flow_one_hot_protocol_axis +launcher=slurm dispatcher.h5ad_data_kwargs.cofactor=3000\
#          callbacks.wandb_project_name="SFC_Goettgens" dispatcher=one_hot_protocol_axis data=one_hot_protocol_axis vf=one_hot_protocol_axis;
