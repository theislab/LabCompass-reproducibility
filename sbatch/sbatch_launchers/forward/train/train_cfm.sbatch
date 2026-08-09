#!/bin/bash

#SBATCH --job-name=train_flow_matching
#SBATCH --output=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/conditional_model/train_flow_matching-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/conditional_model/train_flow_matching-%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00


source ~/.bashrc
ENV_NAME="jupyter"
echo "Using environment: $ENV_NAME"

WDIR=/home/icb/ilia.navosha/expDesign/collab-goettgens-SFC
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

micromamba run -n "$ENV_NAME" python ./scripts/forward/train/train_flow_matching.py