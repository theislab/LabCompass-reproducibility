#!/bin/bash

#SBATCH --job-name=train_flow
#SBATCH --output=logs/train_flow.out
#SBATCH --error=logs/train_flow.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=6
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

## set environment name
ENV_NAME="sc_exp_design";
echo "Using environment  $ENV_NAME";

## hydra error logging
export HYDRA_FULL_ERROR=1;

## define configuration variables
CONFIG_NAME="train_flow"
GOETTGENS_CONFIG_PATH=../config;

## Spectral Flow Cytometry unconditional
# echo "Running experiment on Spectral Flow Cytometry data in the unconditional case";
# conda run -n $ENV_NAME python ./scripts/train_flow.py --config-path $GOETTGENS_CONFIG_PATH --config-name $CONFIG_NAME \
#          callbacks.wandb_project_name="SFC_Goettgens";

## Spectral Flow Cytometry one hot protocols
echo "Running experiment on Spectral Flow Cytometry data with one hot encoded protocols";
conda run -n $ENV_NAME python ./scripts/train_flow.py --config-path $GOETTGENS_CONFIG_PATH --config-name $CONFIG_NAME \
         callbacks.wandb_project_name="SFC_Goettgens" dispatcher=one_hot_protocols data=one_hot_protocols vf=one_hot_protocols;

## Spectral Flow Cytometry one hot protocol axis
# echo "Running experiment on Spectral Flow Cytometry data with one hot encoded protocol axis";
# conda run -n $ENV_NAME python ./scripts/train_flow.py --config-path $GOETTGENS_CONFIG_PATH --config-name $CONFIG_NAME \
#          callbacks.wandb_project_name="SFC_Goettgens" dispatcher=one_hot_protocol_axis data=one_hot_protocol_axis vf=one_hot_protocol_axis;
