#!/bin/bash

#SBATCH --job-name=sweep_flow
#SBATCH --output=../../logs/sweep_flow_morphogen.out
#SBATCH --error=../../logs/sweep_flow_morphogen.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=256G
#SBATCH --nice=0
#SBATCH --time=03-00:00:00

## set environment name
ENV_NAME="sc_exp_design";
echo "Using environment  $ENV_NAME";

## set working directory
WDIR=./scripts;
cd $WDIR;
echo "Current Working Directory $WDIR";

## hydra error logging
export HYDRA_FULL_ERROR=1;

## Sweep CFM channel features
echo "Running sweep on SFC cambridge (perturbation response prediction model)";
echo "Hydra configuration path set to $MORPHOGEN_CONFIG_PATH.";
echo "Hydra configuration name set to $CONFIG_NAME.";
conda run -n $ENV_NAME python ./train_flow_matching.py\
    --multirun\
    +launcher=slurm  \
    +hparams_search=ohe_protocols\
    callbacks.wandb_project_name="SFC_cambridge_sweep_CFM_ohe_protocols";
