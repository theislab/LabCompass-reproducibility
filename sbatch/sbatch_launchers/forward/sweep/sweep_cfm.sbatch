#!/bin/bash

#SBATCH --job-name=sweep_flow_matching
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/conditional_model/job_submissions/sweep_flow_matching-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/conditional_model/job_submissions/sweep_flow_matching-%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00


SAMPLE_REP="X_channel_standardized+X_scatter_standardized"
PROTOCOL_AXIS="protocol_id"
UNIQUE_VALUE=0
CONDITION_MODE="protocol_concat"
USE_NEW_MEASUREMENTS=4
HPARAMSEARCH="in_distribution_big"
OOD_EXPERIMENT=0


# Parse Flags
while [[ $# -gt 0 ]]; do
  case $1 in
    -sr|--sample_rep)
      SAMPLE_REP="$2"
      shift 2
      ;;
    -p|--protocol-axis)
      PROTOCOL_AXIS="$2"
      shift 2
      ;;
    -u|--unique-value)
      UNIQUE_VALUE="$2"
      shift 2
      ;;
    -c|--condition-mode)
      CONDITION_MODE="$2"
      shift 2
      ;;
    -nm|--new-measurements)
      USE_NEW_MEASUREMENTS="$2"
      shift 2
      ;;
    -h|--hparams)
      HPARAMSEARCH="$2"
      shift 2
      ;;
    -o|--ood)
      OOD_EXPERIMENT="$2"
      shift 2
      ;;
  esac
done


echo "Running experiments on sample: $SAMPLE_REP, condition: $CONDITION_MODE, ood col: $PROTOCOL_AXIS, ood val: $UNIQUE_VALUE, hparams: ${HPARAMSEARCH}, use_new_measurements: ${USE_NEW_MEASUREMENTS}"

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1


wandb_name="sweep_cfm_${CONDITION_MODE}_${SAMPLE_REP}_${PROTOCOL_AXIS}_${HPARAMSEARCH}"
paths="hpc_full"
annotation="default"
transforms="default"
if [[ $USE_NEW_MEASUREMENTS -eq 1 ]]; then
    paths="${paths}_new_measurements"
    wandb_name="${wandb_name}_new_measurements"
elif [[ $USE_NEW_MEASUREMENTS -eq 2 ]]; then
    paths="${paths}_bloodplus"
    wandb_name="${wandb_name}_bloodplus"
    annotation="bloodplus"
    transforms="bloodplus"

elif [[ $USE_NEW_MEASUREMENTS -eq 3 ]]; then
    paths="${paths}_bloodplus_old_measurements"
    wandb_name="${wandb_name}_bloodplus_old"
    annotation="bloodplus"
    transforms="bloodplus"

elif [[ $USE_NEW_MEASUREMENTS -eq 4 ]]; then
    paths="${paths}_bloodplus_loop2"
    wandb_name="${wandb_name}_bloodplus_loop2"
    annotation="bloodplus"
    transforms="bloodplus"
fi

split_mode="in-distribution"
if [[ $OOD_EXPERIMENT -eq 1 ]]; then
  split_mode="ood"
fi

# conda run --no-capture-output -n "$ENV_NAME" python ${WDIR}/scripts/forward/train/train_flow_matching.py \
micromamba run -n "$ENV_NAME" python ${WDIR}/scripts/forward/train/train_flow_matching.py \
    --multirun \
    +launcher=slurm \
    +hparams_search="${HPARAMSEARCH}" \
    annotation=${annotation} \
    transforms=${transforms} \
    paths=${paths} \
    vf="default_${CONDITION_MODE}" \
    data="${CONDITION_MODE}" \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    split.mode=${split_mode} \
    split.obs_column='"'"${PROTOCOL_AXIS}"'"' \
    split.unique_value_ids="$UNIQUE_VALUE"
