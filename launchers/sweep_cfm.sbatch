#!/bin/bash

#SBATCH --job-name=sweep_cfm
#SBATCH --output=../logs/sweep_flow_haematopoiesis.log
#SBATCH --error=../logs/sweep_flow_haematopoiesis.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=256G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00


SAMPLE_REP=${1:-"X_channel_standardized+X_scatter_standardized"}
PROTOCOL_AXIS=${2:-"protocol_id"}
UNIQUE_VALUE=${3:-0}
CONDITION_MODE=${4:-"protocol_concat"}

echo "Running experiments on sample: $SAMPLE_REP, condition: $CONDITION_MODE, ood col: $PROTOCOL_AXIS, ood val: $UNIQUE_VALUE"

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=./scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

wandb_name="haematopoiesis_forward_model_new_${CONDITION_MODE}_${SAMPLE_REP}"

conda run --no-capture-output -n "$ENV_NAME" python train_flow_matching.py \
    --multirun \
    +launcher=slurm \
    +hparams_search=default \
    vf="default_${CONDITION_MODE}" \
    data="${CONDITION_MODE}" \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    ood.obs_column='"'"${PROTOCOL_AXIS}"'"' \
    ood.unique_value_ids="$UNIQUE_VALUE" \
    2>&1 | tee "../../logs/sweep_cfm-${CONDITION_MODE}_${SAMPLE_REP}-${PROTOCOL_AXIS}-${UNIQUE_VALUE}.out"
