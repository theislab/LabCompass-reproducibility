#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASE_SBATCH_DIR="$REPO_ROOT/sbatch"

SWEEP_FLAG=0
VALIDATE_FLAG=0
CLASSIFIER_FLAG=0
NEW_MEASUREMENTS_FLAG=4
RERUN_EXPERIMENTS_FLAG=0

# Check for debugging flag
for arg in "$@"; do
    if [[ "$arg" == "--sweep" ]]; then
        SWEEP_FLAG=1
    fi
    if [[ "$arg" == "--validate" ]]; then
        VALIDATE_FLAG=1
    fi
    if [[ "$arg" == "--classification" ]]; then
        CLASSIFIER_FLAG=1
    fi
done

SAMPLE_REPS=(
  # "X_channel"
  # "X_channel_standardized"
  # "X_scatter"
  # "X_scatter_standardized"
  # "X_channel+X_scatter"
  # "X_channel+X_scatter_standardized"
  # "X_channel_standardized+X_scatter"
  # "X_channel_standardized+X_scatter_standardized"
  "X_channel+X_scatter_zca_whitened"
  "X_channel+X_scatter_pca_whitened"
  "X_channel+X_scatter_cholesky_whitened"
  # "X_pca"
  # "X_pca+X_scatter"
  # "X_pca+X_scatter_standardized"
)

echo "${SAMPLE_REPS[@]}"

declare -A PROTOCOL_AXES_TO_NUNIQUE=(
  ["tpo_[ng_ml]"]=1
  # ["tpo_[ng_ml]"]=5
  # ["um171_[nm]"]=6
  # ["um729_[µm]"]=4
  # ["scf_[ng_ml]"]=3
  # ["butyzamide_[nm]"]=2
  # ["days_of_culture"]=6
  # ["il3_[ng_ml]"]=5
  # ["dummy_col"]=1
  # ["retinoic_acid_[µm]"]=3
  # ["gm-csf_[ng_ml]"]=3
  # ["sr1_[nm]"]=4
  # ["ldl_[ng_ml]"]=6
  # ["o2_[%]"]=2
  # ["protocol_id"]=116
)

# CONDITIONING_MODE=("unconditional" "ohe_protocols" "ohe_protocols_axes" "protocol_concat")
# CONDITIONING_MODE=("ohe_protocols" "ohe_protocols_axes")
# CONDITIONING_MODE=("protocol_concat" ohe_protocols_axes)
# CONDITIONING_MODE=("ohe_protocols_axes")
# CONDITIONING_MODE=("protocol_concat", "ohe_protocols_axes")
CONDITIONING_MODE=("protocol_concat")

RUNS_TO_RERUN=(
    "neat-wildflower-3"
    "dark-cloud-16"
    "solar-pine-15"
    "fine-gorge-23"
)

for sample_rep in "${SAMPLE_REPS[@]}"; do
  if [[ $CLASSIFIER_FLAG -eq 1 ]]; then
    if [[ $SWEEP_FLAG -eq 1 ]]; then
      echo "Running Sweep (Target Prediction Model)"
      sbatch ${BASE_SBATCH_DIR}/sbatch_launchers/forward/sweep/sweep_target_prediction_model.sbatch \
        --sample_rep "$sample_rep" \
        --new-measurements "$NEW_MEASUREMENTS_FLAG"
    else
      echo "Running base training (Target Prediction Model)"
      sbatch launchers/train_target_prediction_model.sbatch \
        --sample_rep "$sample_rep" \
        --new-measurements "$NEW_MEASUREMENTS_FLAG"
    fi
  else
    for conditioning_mode in "${CONDITIONING_MODE[@]}"; do
      if [[ $RERUN_EXPERIMENTS_FLAG -eq 1 ]]; then
        echo "Using sample rep $sample_rep and conditioning mode $conditioning_mode"
        echo "Rerunning experiment"
        for run_name in "${RUNS_TO_RERUN[@]}"; do
          echo "Launching Experiment for ${run_name}"
          sbatch "${BASE_SBATCH_DIR}/sbatch_launchers/forward/train/train_cfm.sbatch" \
            --sample_rep "$sample_rep" \
            --condition-mode "$conditioning_mode" \
            --rerun-experiment "$RERUN_EXPERIMENTS_FLAG" \
            --run-name "$run_name" \
            --new-measurements "$NEW_MEASUREMENTS_FLAG"
        done
      else
        for protocol_axis in "${!PROTOCOL_AXES_TO_NUNIQUE[@]}"; do
          nunique=${PROTOCOL_AXES_TO_NUNIQUE[$protocol_axis]}
            
          for unique_val in $(seq 0 $((nunique - 1))); do
            echo "Using sample rep $sample_rep and conditioning mode $conditioning_mode"
            echo "Running for protocol axis: $protocol_axis with unique IDs: $unique_val"
            
            if [[ $SWEEP_FLAG -eq 1 ]]; then
              echo "Running Sweep"
              sbatch "${BASE_SBATCH_DIR}/sbatch_launchers/forward/sweep/sweep_cfm.sbatch" \
                --sample_rep "$sample_rep" \
                --protocol-axis "$protocol_axis" \
                --unique-value "$unique_val" \
                --condition-mode "$conditioning_mode" \
                --new-measurements "$NEW_MEASUREMENTS_FLAG" \
                --rerun-experiment "$RERUN_EXPERIMENTS_FLAG"
            else
              echo "Running base training"
              sbatch "${BASE_SBATCH_DIR}/sbatch_launchers/forward/train/train_cfm.sbatch" \
                --sample_rep "$sample_rep" \
                --protocol-axis "$protocol_axis" \
                --unique-value "$unique_val" \
                --condition-mode "$conditioning_mode" \
                --new-measurements "$NEW_MEASUREMENTS_FLAG" \
                --rerun-experiment "$RERUN_EXPERIMENTS_FLAG"
            fi
          done
        done
      fi
    done
  fi
done