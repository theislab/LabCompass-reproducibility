#!/bin/bash

#SBATCH --job-name=train_cfm_ohe_protocol_axes
#SBATCH --output=../logs/train_flow_haematopoiesis_ohe_protocol_axes.out
#SBATCH --error=../logs/train_flow_haematopoiesis_ohe_protocol_axes.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00


SAMPLE_REP=${1:-all}
PROTOCOL_AXIS=${2:-all}
UNIQUE_VALUE=${3:-all}

echo "Running experiments on $SAMPLE_REP $PROTOCOL_AXIS, $UNIQUE_VALUE"

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=./scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

wandb_name="haematopoiesis_sweep_forward_model_ohe_protocols_axes_${SAMPLE_REP}"

conda run --no-capture-output -n "$ENV_NAME" python train_flow_matching.py \
    transforms.sample_rep="$SAMPLE_REP" \
    vf=default_ohe_protocols_axes \
    data=ohe_protocols_axes \
    callbacks.wandb_project_name="$wandb_name" \
    ood.obs_column='"'"${PROTOCOL_AXIS}"'"' \
    ood.unique_value_ids="$UNIQUE_VALUE"
