#!/bin/bash

#SBATCH --job-name=sweep_classifier
#SBATCH --output=../logs/sweep_classifier.log
#SBATCH --error=../logs/sweep_classifier.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_priority
#SBATCH --mem=50G
#SBATCH --nice=0
#SBATCH --time=0-02:00:00


SAMPLE_REP=${1:-"X_channel_standardized+X_scatter_standardized"}

echo "Running experiments on sample: $SAMPLE_REP"

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

wandb_name="sweep_classifier_${SAMPLE_REP}"
echo "Wandb project ${wandb_name}"

conda run --no-capture-output -n "$ENV_NAME" python ${WDIR}/scripts/forward/train/train_target_prediction_model.py \
    --multirun \
    +launcher=slurm_preentible \
    +hparams_search=default \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    2>&1 | tee "../../logs/sweep_classifier-${SAMPLE_REP}.out"
