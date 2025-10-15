#!/bin/bash

#SBATCH --job-name=train_cfm
#SBATCH --output=../logs/train_flow_haematopoiesis.out
#SBATCH --error=../logs/train_flow_haematopoiesis.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=01-00:00:00

# =======================
# ARGUMENT HANDLING
# =======================
SAMPLE_ARGS=${1:-all}
CONDITION_ARGS=${2:-all}
DEBUG_FLAG=0

# Check for debugging flag
for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        DEBUG_FLAG=1
    fi
done

# =======================
# ENVIRONMENT SETUP
# =======================
ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"

WDIR=./scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

# =======================
# VARIABLE DEFINITIONS
# =======================
declare -A SAMPLE_REPS=(
  ["channel"]="X_channel"
  ["pca"]="X_pca"
  ["scatter"]="X_scatter"
  ["channel_concat"]="X_channel_concat"
  ["pca_concat"]="X_pca_concat"
)

declare -A CONDITION_MODES=(
  ["unconditional"]="unconditional"
  ["protocol_concat"]="protocol_concat"
  ["ohe_protocols"]="ohe_protocols"
  ["ohe_protocols_axes"]="ohe_protocols_axes"
)

# =======================
# FUNCTION DEFINITION
# =======================
run_case() {
  local sample_name=$1
  local sample_rep=$2
  local condition_name=$3
  local condition_mode=$4

  # Set arguments dynamically based on debug flag
  if [[ $DEBUG_FLAG -eq 1 ]]; then
    training="debug"
    paths="hpc_subset"
    cond_mode="$condition_mode"
    wandb_name="haematopoiesis_forward_model_debug_${sample_rep}"
  else
    training="default"
    paths="hpc_full"
    cond_mode="$condition_mode"
    wandb_name="haematopoiesis_forward_model_${condition_mode}_${sample_rep}"
  fi

  echo "Running script in mode ${training}"
  echo "Sample representation: ${sample_name}"
  echo "Conditioning mode ${cond_mode}"
  echo "Running on wandb project ${wandb_name}"

  # Single conda run command
  conda run --no-capture-output -n "$ENV_NAME" python train_flow_matching.py \
    ---multirun \
    +launcher=slurm \
    +hparams="$cond_mode"\
    ood="$cond_mode" \
    training="$training" \
    paths="$paths" \
    transforms.sample_rep="$sample_rep" \
    vf="default_${cond_mode}" \
    data="${cond_mode}" \
    callbacks.wandb_project_name="$wandb_name"
}

# =======================
# MAIN EXECUTION
# =======================
for sample in "${!SAMPLE_REPS[@]}"; do
  if [[ "$SAMPLE_ARGS" == "all" || " $SAMPLE_ARGS " =~ " $sample " ]]; then
    for condition in "${!CONDITION_MODES[@]}"; do
      if [[ "$CONDITION_ARGS" == "all" || " $CONDITION_ARGS " =~ " $condition " ]]; then
        run_case "$sample" "${SAMPLE_REPS[$sample]}" "$condition" "${CONDITION_MODES[$condition]}"
      fi
    done
  fi
done
