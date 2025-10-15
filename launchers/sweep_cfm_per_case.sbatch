#!/bin/bash

#SBATCH --job-name=train_cfm
#SBATCH --output=../logs/sweep_flow_haematopoiesis.out
#SBATCH --error=../logs/sweep_flow_haematopoiesis.err
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
  ["protocol_concat"]="protocol_concat"
  ["ohe_protocols"]="ohe_protocols"
  ["ohe_protocols_axes"]="ohe_protocols_axes"
)

declare -A PROTOCOL_AXES_TO_NUNIQUE=(
  ["gm-csf_[ng_ml]"]=3
  ["tpo_[ng_ml]"]=5
  ["sr1_[nm]"]=4
  ["um171_[nm]"]=6
  ["butyzamide_[nm]"]=2
  ["retinoic_acid_[µm]"]=3
  ["ldl_[ng_ml]"]=6
  ["il3_[ng_ml]"]=5
  ["o2_[%]"]=2
  ["days_of_culture"]=6
)

declare -A CONDITION_MODE_TO_UNIQUE_VALUES=(
  ["ohe_protocols"]="$(seq -s, 0 115)"
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
    wandb_name="haematopoiesis_sweep_forward_model_debug_${sample_rep}"
  else
    training="default"
    paths="hpc_full"
    cond_mode="$condition_mode"
    wandb_name="haematopoiesis_sweep_forward_model_${condition_mode}_${sample_rep}"
  fi

  # Handle ohe_protocols_axes separately
  if [[ "$condition_name" == "ohe_protocols_axes" ]]; then
    for protocol_axis in "${!PROTOCOL_AXES_TO_NUNIQUE[@]}"; do
      nunique=${PROTOCOL_AXES_TO_NUNIQUE[$protocol_axis]}
      unique_ids=$(seq 0 $((nunique - 1)))

      echo "Running for protocol axis: $protocol_axis with unique IDs: $unique_ids"

      # Redirect output to separate logs to avoid collisions
      LOG_OUT="../logs/${wandb_name}_${protocol_axis}.out"
      LOG_ERR="../logs/${wandb_name}_${protocol_axis}.err"
  
      conda run --no-capture-output -n "$ENV_NAME" python train_flow_matching.py \
        --multirun \
        +launcher=slurm \
        +hparams="$condition_mode" \
        # ood="$condition_mode" \
        training="$training" \
        paths="$paths" \
        transforms.sample_rep="$sample_rep" \
        vf="default_${condition_mode}" \
        data="${condition_mode}" \
        callbacks.wandb_project_name="$wandb_name" \
        obs_column="$protocol_axis" \
        unique_value_ids="$unique_ids"\
        > "$LOG_OUT" 2> "$LOG_ERR" &
    done
  else
    echo "Running script in mode ${training}"
    echo "Sample representation: ${sample_name}"
    echo "Conditioning mode ${condition_mode}"
    echo "Running on wandb project ${wandb_name}"

    conda run --no-capture-output -n "$ENV_NAME" python train_flow_matching.py \
      --multirun \
      +launcher=slurm \
      +hparams="$condition_mode" \
      ood="$condition_mode" \
      training="$training" \
      paths="$paths" \
      transforms.sample_rep="$sample_rep" \
      vf="default_${condition_mode}" \
      data="${condition_mode}" \
      callbacks.wandb_project_name="$wandb_name"
  fi
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
