#!/bin/bash

#SBATCH --job-name=validate_cfm
#SBATCH --output=../logs/validate_cfm_haematopoiesis.log
#SBATCH --error=../logs/validate_cfm_haematopoiesis.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00


PROJECT_NAME=${1:-all}
PROTOCOL_AXIS=${2:-"protocol_id"}
UNIQUE_VALUE=${3:-0}

echo "Running experiments on $PROJECT_NAME, $PROTOCOL_AXIS, $UNIQUE_VALUE."

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=./scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1


conda run --no-capture-output -n "$ENV_NAME" python validate_flow_matching.py \
    run.project="${PROJECT_NAME}" \
    ood.obs_column='"'"${PROTOCOL_AXIS}"'"' \
    ood.unique_value_ids="$UNIQUE_VALUE" \
    2>&1 | tee "../../logs/validate_cfm-${PROJECT_NAME}_${PROTOCOL_AXIS}-${UNIQUE_VALUE}.out"
