#!/bin/bash

#SBATCH --job-name=sweep_classifier
#SBATCH --output=sweep_classifier-%j.log
#SBATCH --error=sweep_classifier-%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
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



SAMPLE_REP="X_channel_standardized+X_scatter_standardized"
USE_NEW_MEASUREMENTS=4
USE_NEW_CAT=0


# Parse Flags
while [[ $# -gt 0 ]]; do
  case $1 in
    -sr|--sample_rep)
      SAMPLE_REP="$2"
      shift 2
      ;;
    -nm|--new-measurements)
      USE_NEW_MEASUREMENTS="$2"
      shift 2
      ;;
    -nc|--new-cat)
      USE_NEW_CAT="$2"
      shift 2
      ;;
  esac
done


echo "Running experiments on sample: $SAMPLE_REP"

# ENV_NAME="sc_exp_design"
ENV_NAME="sc_exp_design_12"
echo "Using environment: $ENV_NAME"

WDIR=$REPO_ROOT

cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

wandb_name="sweep_classifier_${CONDITION_MODE}_${SAMPLE_REP}_${PROTOCOL_AXIS}_${HPARAMSEARCH}"
paths="hpc_subset"
transforms="default"
annotation="default"
model="cell_type_default.yaml"
data="cell_type_default"
if [[ $USE_NEW_MEASUREMENTS -eq 1 ]]; then
    paths="${paths}_new_measurements"
    wandb_name="${wandb_name}_new_measurements"
elif [[ $USE_NEW_MEASUREMENTS -eq 2 ]]; then
    paths="${paths}_bloodplus"
    wandb_name="${wandb_name}_bloodplus"
    transforms="bloodplus"
    model="cell_type_bloodplus"
elif [[ $USE_NEW_MEASUREMENTS -eq 3 ]]; then
    paths="${paths}_bloodplus_old_measurements"
    wandb_name="${wandb_name}_bloodplus_old"
    transforms="bloodplus"
    model="cell_type_bloodplus"

elif [[ $USE_NEW_MEASUREMENTS -eq 4 ]]; then
    paths="${paths}_bloodplus_loop2"
    # wandb_name="bloodplus_loop2" 
    annotation="bloodplus"
    transforms="bloodplus"
    data="cell_type_leiden"
    wandb_name="bloodplus_loop2_classifier"
    model="cell_type_bloodplus_leiden"

    if [[ $USE_NEW_CAT -eq 1 ]]; then
        data="cell_type_leiden_new_cat"
        wandb_name="bloodplus_loop2_classifier_new_cat"
        model="cell_type_bloodplus_leiden_new_cat"
    fi

fi

echo "Wandb project ${wandb_name}"
echo "paths ${paths}"

# micromamba run -n "$ENV_NAME" python ./scripts/forward/train/train_target_prediction_model.py \
conda run --no-capture-output -n "$ENV_NAME" python ./scripts/forward/train/train_target_prediction_model.py \
    --multirun \
    +launcher=slurm \
    +hparams_search=small \
    data=${data} \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    paths="${paths}" \
    transforms="${transforms}" \
    model="${model}"

