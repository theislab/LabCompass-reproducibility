#!/bin/bash

#SBATCH --job-name=sweep_flow_matching
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/cell_type_classifier/job_submissions/sweep_clf_loops-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/cell_type_classifier/job_submissions/sweep_clf_loops-%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00

SAMPLE_REP="X_channel_standardized+X_scatter_standardized"
LOOP="3"
HPARAMSEARCH="small"
ENV_MANAGER="micromamba"
ENV_NAME="sc_exp_design"

# Parse Flags
while [[ $# -gt 0 ]]; do
  case $1 in
    -sr|--sample_rep)
      SAMPLE_REP="$2"
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
    -em|--env-manager)
      ENV_MANAGER="$2"
      shift 2
      ;;
    -en|--env-name)
      ENV_NAME="$2"
      shift 2
      ;;
  esac
done


echo "Running experiments on sample: $SAMPLE_REP, hparams: ${HPARAMSEARCH}, loop: ${LOOP}, env manager: ${ENV_MANAGER}, env name: ${ENV_NAME}"

WDIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

data="cell_type_default"
wandb_name="sweep_classifier"
paths="loop2_replicate"
transforms="bloodplus"

case $LOOP in
  2)
    data="cell_type_reannot_final"
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
    ;;
  *)
    echo "Unknown LOOP value: $LOOP"
    exit 1
    ;;
esac


case $ENV_MANAGER in
  "conda")
    cmd="conda run --no-capture-output -n ${ENV_NAME} python ${WDIR}/scripts/forward/train/train_target_prediction_model.py"
    ;;
  "micromamba")
    cmd="micromamba run -n ${ENV_NAME} python ${WDIR}/scripts/forward/train/train_target_prediction_model.py"
    ;;
  *)
    echo "Unknown env manager value: $ENV_MANAGER"
    exit 1
    ;;
esac

$cmd --multirun \
    +launcher=slurm \
    +hparams_search=${HPARAMSEARCH} \
    data=${data} \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    paths="${paths}" \
    transforms="${transforms}" \
    model="${model}"