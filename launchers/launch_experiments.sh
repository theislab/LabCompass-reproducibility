#!/bin/bash

SWEEP_FLAG=0
VALIDATE_FLAG=0
# Check for debugging flag
for arg in "$@"; do
    if [[ "$arg" == "--sweep" ]]; then
        SWEEP_FLAG=1
    fi
    if [[ "$arg" == "--validate" ]]; then
        VALIDATE_FLAG=1
    fi
done

SAMPLE_REPS=(
   "X_channel"
  # "X_channel_standardized"
  # "X_scatter"
  # "X_scatter_standardized"
  # "X_channel+X_scatter"
  #  "X_channel+X_scatter_standardized"
  # "X_channel_standardized+X_scatter"
  # "X_channel_standardized+X_scatter_standardized"
  #  "X_pca"
  # "X_pca+X_scatter"
  #  "X_pca+X_scatter_standardized"
)

declare -A PROTOCOL_AXES_TO_NUNIQUE=(
  # ["gm-csf_[ng_ml]"]=3
  # ["tpo_[ng_ml]"]=5
  # ["sr1_[nm]"]=4
  # ["um171_[nm]"]=6
  # ["butyzamide_[nm]"]=2
  # ["retinoic_acid_[µm]"]=3
  # ["ldl_[ng_ml]"]=6
  # ["il3_[ng_ml]"]=5
  ["o2_[%]"]=2
  # ["days_of_culture"]=6
  # ["protocol_id"]=116
)

# CONDITIONING_MODE=("unconditional" "ohe_protocols" "ohe_protocols_axes" "protocol_concat")
# CONDITIONING_MODE=("ohe_protocols" "ohe_protocols_axes")
CONDITIONING_MODE=("protocol_concat")

for sample_rep in "${SAMPLE_REPS[@]}"; do
  for protocol_axis in "${!PROTOCOL_AXES_TO_NUNIQUE[@]}"; do
    rep=${SAMPLE_REPS[$sample_rep]}
    nunique=${PROTOCOL_AXES_TO_NUNIQUE[$protocol_axis]}
    for unique_val in $(seq 0 $((nunique - 1))); do
      for conditioning_mode in "${CONDITIONING_MODE[@]}"; do
        echo "Using sample rep ${rep} and conditioning mode ${conditioning_mode}"
        echo "Running for protocol axis: $protocol_axis with unique IDs: $unique_val"
        if [[ $SWEEP_FLAG -eq 1 ]]; then
          sbatch launchers/sweep_cfm.sbatch $rep $protocol_axis $unique_val $conditioning_mode
        fi
        if [[ $VALIDATE_FLAG -eq 1 ]]; then
          sbatch launchers/validate_cfm.sbatch $rep $protocol_axis $unique_val $conditioning_mode

        else
          sbatch launchers/train_cfm.sbatch $rep $protocol_axis $unique_val $conditioning_mode
        fi
      done
    done
  done
done
