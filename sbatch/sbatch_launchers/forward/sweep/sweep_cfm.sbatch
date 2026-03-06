#!/bin/bash

#SBATCH --job-name=sweep_flow_matching
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/conditional_model/job_submissions/sweep_flow_matching-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/conditional_model/job_submissions/sweep_flow_matching-%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00


SAMPLE_REP=${1:-"X_channel_standardized+X_scatter_standardized"}
PROTOCOL_AXIS=${2:-"protocol_id"}
UNIQUE_VALUE=${3:-0}
CONDITION_MODE=${4:-"protocol_concat"}
USE_NEW_MEASUREMENTS=${5:-1}
HPARAMSEARCH=${6:-"in_distribution_big"}


echo "Running experiments on sample: $SAMPLE_REP, condition: $CONDITION_MODE, ood col: $PROTOCOL_AXIS, ood val: $UNIQUE_VALUE, hparams: ${HPARAMSEARCH}, use_new_measurements: ${USE_NEW_MEASUREMENTS}"

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1


wandb_name="sweep_cfm_${CONDITION_MODE}_${SAMPLE_REP}_${PROTOCOL_AXIS}_${HPARAMSEARCH}"
paths="hpc_full"
if [[ $USE_NEW_MEASUREMENTS -eq 1 ]]; then
    paths="${paths}_new_measurements"
    wandb_name="${wandb_name}_new_measurements"
fi


conda run --no-capture-output -n "$ENV_NAME" python ${WDIR}/scripts/forward/train/train_flow_matching.py \
    --multirun \
    +launcher=slurm \
    +hparams_search="${HPARAMSEARCH}" \
    paths=${paths} \
    vf="default_${CONDITION_MODE}" \
    data="${CONDITION_MODE}" \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    split.obs_column='"'"${PROTOCOL_AXIS}"'"' \
    split.unique_value_ids="$UNIQUE_VALUE"
