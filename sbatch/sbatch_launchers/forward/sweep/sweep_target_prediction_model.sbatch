#!/bin/bash

#SBATCH --job-name=sweep_classifier
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/cell_type_classifier/job_submissions/sweep_classifier-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/cell_type_classifier/job_submissions/sweep_classifier-%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_priority
#SBATCH --mem=50G
#SBATCH --nice=0
#SBATCH --time=0-02:00:00


SAMPLE_REP=${1:-"X_channel_standardized+X_scatter_standardized"}
USE_NEW_MEASUREMENTS=${2:-1}

echo "Running experiments on sample: $SAMPLE_REP"

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

wandb_name="sweep_classifier_${SAMPLE_REP}"
wandb_name="sweep_cfm_${CONDITION_MODE}_${SAMPLE_REP}_${PROTOCOL_AXIS}_${HPARAMSEARCH}"
paths="hpc_subset"
if [[ $USE_NEW_MEASUREMENTS -eq 1 ]]; then
    paths="${paths}_new_measurements"
    wandb_name="${wandb_name}_new_measurements"
fi
echo "Wandb project ${wandb_name}"

conda run --no-capture-output -n "$ENV_NAME" python ${WDIR}/scripts/forward/train/train_target_prediction_model.py \
    --multirun \
    +launcher=slurm_preentible \
    +hparams_search=default \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name"
