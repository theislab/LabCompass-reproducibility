#!/bin/bash

#SBATCH --job-name=train_classifier
#SBATCH --output=train_classifier-%j.log
#SBATCH --error=train_classifier-%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00


# SLURM copies the batch script to a node-local spool dir, so BASH_SOURCE does
# not point into the repo. Recover the submitted path from the job record.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}' | awk '{print $1}')"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if [ ! -d "$REPO_ROOT/scripts" ]; then
    echo "ERROR: REPO_ROOT resolved to '$REPO_ROOT', which is not the repository." >&2
    exit 1
fi


SAMPLE_REP=${1:-"X_channel_standardized+X_scatter_standardized"}

echo "Running experiments on sample: $SAMPLE_REP"

ENV_NAME="labcompass"
echo "Using environment: $ENV_NAME"

WDIR=$REPO_ROOT
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

wandb_name="train_classifier_${SAMPLE_REP}"

LOG_DIR="$REPO_ROOT/project_folder/output/logs/forward"
mkdir -p "$LOG_DIR"

conda run --no-capture-output -n "$ENV_NAME" python ./scripts/forward/train/train_target_prediction_model.py \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    2>&1 | tee "$LOG_DIR/train_classifier-${SAMPLE_REP}.out"
