#!/bin/bash

#SBATCH --job-name=train_classifier
#SBATCH --output=../logs/train_classifier.log
#SBATCH --error=../logs/train_classifier.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00


SAMPLE_REP=${1:-"X_channel_standardized+X_scatter_standardized"}

echo "Running experiments on sample: $SAMPLE_REP"

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=./scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

wandb_name="train_classifier_${SAMPLE_REP}"

conda run --no-capture-output -n "$ENV_NAME" python train_target_prediction_model.py \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    2>&1 | tee "../../logs/train_classifier-${SAMPLE_REP}.out"
