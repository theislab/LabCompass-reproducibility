#!/bin/bash

#SBATCH --job-name=train_classifier
#SBATCH --output=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/cell_type_classifier/train_classifier-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/cell_type_classifier/train_classifier-%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

source ~/.bashrc

SAMPLE_REP=${1:-"X_train_pca"}

echo "Running experiments on sample: $SAMPLE_REP"

ENV_NAME="jupyter"
echo "Using environment: $ENV_NAME"

WDIR=../../../..
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

micromamba run -n "$ENV_NAME" python ./scripts/forward/train/train_target_prediction_model_RA_4.py
