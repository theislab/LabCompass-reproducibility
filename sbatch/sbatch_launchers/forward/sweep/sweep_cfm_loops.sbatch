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
LOOP="2"
HPARAMSEARCH="in_distribution_big"
ENV_MANAGER="micromamba"

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
    -l|--loop)
      LOOP="$2"
      shift 2
      ;;
    -h|--hparams)
      HPARAMSEARCH="$2"
      shift 2
      ;;
  esac
done


echo "Running experiments on sample: $SAMPLE_REP, condition: $CONDITION_MODE, ood col: $PROTOCOL_AXIS, ood val: $UNIQUE_VALUE, hparams: ${HPARAMSEARCH}, loop: ${LOOP}"

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1


wandb_name="sweep_flow_matching"
paths="hpc_full"
annotation="bloodplus"
transforms="bloodplus"
split_mode="in-distribution"
vf="default_${CONDITION_MODE}"

case $LOOP in
  1)
    paths="loop1"
    wandb_name="${wandb_name}_loop1"
    ;;
  2)
    paths="loop2_replicate"
    wandb_name="${wandb_name}_loop2"
    ;;
  2.5)
    paths="loop2p5_replicate"
    wandb_name="${wandb_name}_loop2p5"
    ;;
  3)
    paths="loop3_replicate"
    wandb_name="${wandb_name}_loop3"
    annotation="bloodplus_loop3"
    ;;
  *)
    echo "Unknown LOOP value: $LOOP"
    exit 1
    ;;
esac


micromamba run -n "$ENV_NAME" python ${WDIR}/scripts/forward/train/train_flow_matching.py \
# conda run --no-capture-output -n "$ENV_NAME" python ${WDIR}/scripts/forward/train/train_flow_matching.py \
    --multirun \
    +launcher=slurm \
    +hparams_search="${HPARAMSEARCH}" \
    annotation=${annotation} \
    transforms=${transforms} \
    paths=${paths} \
    vf="${vf}" \
    data="${CONDITION_MODE}" \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    split.mode=${split_mode} \
    split.obs_column='"'"${PROTOCOL_AXIS}"'"' \
    split.unique_value_ids="$UNIQUE_VALUE"
