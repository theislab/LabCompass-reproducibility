#!/bin/bash

#SBATCH --job-name=sweep_flow_ohe_protocols
#SBATCH --output=../logs/sweep_flow_haematopoiesis_ohe_protocols.out
#SBATCH --error=../logs/sweep_flow_haematopoiesis_ohe_protocols.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=256G
#SBATCH --nice=0
#SBATCH --time=01-00:00:00

## set environment name
ENV_NAME="sc_exp_design";
echo "Using environment  $ENV_NAME";

## set working directory
WDIR=./scripts;
cd $WDIR;
echo "Current Working Directory $WDIR";

## hydra error logging
export HYDRA_FULL_ERROR=1;

## Sweep CFM channel features (OHE protocols)
echo "Running sweep on SFC cambridge OHE Protocols (perturbation response prediction model)";
conda run -n $ENV_NAME python ./train_flow_matching.py\
    --multirun\
    +launcher=slurm  \
    +hparams_search=ohe_protocols\
    callbacks.wandb_project_name="SFC_cambridge_sweep_CFM_ohe_protocols";

## Sweep CFM channel features (OHE protocols)
# echo "Running sweep on SFC cambridge OHE Protocols (perturbation response prediction model)";
# conda run -n $ENV_NAME python ./train_flow_matching.py\
#     --multirun\
#     +launcher=slurm  \
#     +hparams_search=ohe_protocols \
#     ood.unique_value_ids=$(seq -s, 0 115) \
#     callbacks.wandb_project_name="SFC_cambridge_sweep_CFM_ohe_protocols_axes";
