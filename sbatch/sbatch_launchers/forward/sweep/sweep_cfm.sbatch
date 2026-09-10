#!/bin/bash

#SBATCH --job-name=sweep_flow_matching
#SBATCH --output=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/conditional_model/flow_matching/multiruns/sweep_flow_matching-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/conditional_model/flow_matching/multiruns/sweep_flow_matching-%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

source ~/.bashrc

SAMPLE_REP="X_train_pca"

echo "Running experiments on sample: $SAMPLE_REP"

ENV_NAME="jupyter"
HPARAMSEARCH="in_distribution_big"
echo "Using environment: $ENV_NAME"

WDIR=/home/icb/ilia.navosha/expDesign/collab-goettgens-SFC
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

micromamba run -n "$ENV_NAME" python ${WDIR}/scripts/forward/train/train_flow_matching.py \
    --multirun \
    +launcher=slurm \
    +hparams_search="${HPARAMSEARCH}" 
