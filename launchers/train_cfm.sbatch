#!/bin/bash

#SBATCH --job-name=train_cfm
#SBATCH --output=../logs/train_flow_haematopoiesis.out
#SBATCH --error=../logs/train_flow_haematopoiesis.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00


SAMPLE_REP=${1:-all}
PROTOCOL_AXIS=${2:-all}
UNIQUE_VALUE=${3:-all}
CONDITION_MODE=${4:-all}

echo "Running experiments on $SAMPLE_REP, $CONDITION_MODE, $PROTOCOL_AXIS, $UNIQUE_VALUE."

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=./scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

wandb_name="haematopoiesis_forward_model_${CONDITION_MODE}_${SAMPLE_REP}"

conda run --no-capture-output -n "$ENV_NAME" python train_flow_matching.py \
    vf="default_${CONDITION_MODE}" \
    data="${CONDITION_MODE}" \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    ood.obs_column='"'"${PROTOCOL_AXIS}"'"' \
    ood.unique_value_ids="$UNIQUE_VALUE" \
    2>&1 | tee "../../logs/train_cfm-${CONDITION_MODE}_${SAMPLE_REP}-${PROTOCOL_AXIS}-${UNIQUE_VALUE}.out"
