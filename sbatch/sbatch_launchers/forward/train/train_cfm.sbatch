#!/bin/bash

#SBATCH --job-name=train_flow_matching
#SBATCH --output=train_flow_matching-%j.log
#SBATCH --error=train_flow_matching-%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00
#SBATCH --constraint="h100_80gb"

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
PROTOCOL_AXIS="protocol_id"
UNIQUE_VALUE=0
CONDITION_MODE="protocol_concat"
RERUN_EXP=1
NEW_MEASUREMENTS_FLAG=0
RUN_NAME="no-name"
ENV_NAME="sc_exp_design"

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
    -rr|--rerun-experiment)
      RERUN_EXP="$2"
      shift 2
      ;;
    -rn|--run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    -nm|--new-measurements)
      NEW_MEASUREMENTS_FLAG="$2"
      shift 2
      ;;
    -e|--env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done


wandb_name="sweep_cfm_${CONDITION_MODE}_${SAMPLE_REP}_${PROTOCOL_AXIS}_${HPARAMSEARCH}"
paths="hpc_full"
if [[ $NEW_MEASUREMENTS_FLAG -eq 1 ]]; then
    paths="${paths}_new_measurements"
    wandb_name="${wandb_name}_new_measurements"
fi

if [[ $RERUN_EXP -eq 1 ]]; then
    vf=$RUN_NAME
    training=$RUN_NAME
    echo "Rerunning experiment ${RUN_NAME} -> vf: ${vf}, training: ${training}"
else
    vf="default_${CONDITION_MODE}"
    training="default"
    echo "Launching default experiment -> vf: ${vf}, training: ${training}"
fi


WDIR=$REPO_ROOT
cd "$WDIR" || exit 1

export HYDRA_FULL_ERROR=1

wandb_name="train_cfm_${CONDITION_MODE}_${SAMPLE_REP}"

echo "--- Job Configuration ---"
echo "Rerun Experiments: ${RERUN_EXP}"
echo "Run Name: ${RUN_NAME}"
echo "Sample Representation: ${SAMPLE_REP}"
echo "OOD Protocol Axis: ${PROTOCOL_AXIS}"
echo "OOD Unique Value: ${UNIQUE_VALUE}"
echo "Conditioning Mode: ${CONDITION_MODE}"
echo "New Measurements Flag: ${NEW_MEASUREMENTS_FLAG}"
echo "WandB Name: ${wandb_name}"
echo "Paths Configuration: ${paths}"
echo "VF Configuration: ${vf}"
echo "Train Configuration: ${training}"
echo "Environment: ${ENV_NAME}"
echo "-------------------------"


conda run --no-capture-output -n "$ENV_NAME" python ${WDIR}/scripts/forward/train/train_flow_matching.py \
    paths=${paths} \
    vf="${vf}" \
    callbacks.wandb_project_name="$wandb_name" \
    data='"'"${CONDITION_MODE}"'"' \
    data.sample_rep="$SAMPLE_REP" \
    callbacks.wandb_project_name="$wandb_name" \
    training="${training}" \
    split.obs_column='"'"${PROTOCOL_AXIS}"'"' \
    split.unique_value_ids="$UNIQUE_VALUE" \
    2>&1 | tee "../../logs/train_cfm-${CONDITION_MODE}_${SAMPLE_REP}-${PROTOCOL_AXIS}-${UNIQUE_VALUE}.out"
