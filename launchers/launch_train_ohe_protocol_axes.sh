#!/bin/bash

SWEEP_FLAG=0
# Check for debugging flag
for arg in "$@"; do
    if [[ "$arg" == "--sweep" ]]; then
        SWEEP_FLAG=1
    fi
done

declare -A SAMPLE_REPS=(
 ["channel"]="X_channel"
 ["pca"]="X_pca"
 ["scatter"]="X_scatter"
 ["channel_concat"]="X_channel_concat"
 ["pca_concat"]="X_pca_concat"
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
 ["protocol_id"]=116
)

CONDITIONING_MODE=("unconditional" "ohe_protocols" "ohe_protocols_axes" "protocols_concat")

for sample_rep in "${!SAMPLE_REPS[@]}"; do
  for protocol_axis in "${!PROTOCOL_AXES_TO_NUNIQUE[@]}"; do
    rep=${SAMPLE_REPS[$sample_rep]}
    nunique=${PROTOCOL_AXES_TO_NUNIQUE[$protocol_axis]}
    for unique_val in $(seq 0 $((nunique - 1))); do
      for conditioning_mode in "${CONDITIONING_MODE[@]}"; do
        echo "Using sample rep ${rep} and conditioning mode ${conditioning_mode}"
        echo "Running for protocol axis: $protocol_axis with unique IDs: $unique_val"
        if [[ $SWEEP_FLAG -eq 1 ]]; then
          sbatch launchers/train_ohe_protocol_axes.sbatch $rep $protocol_axis $unique_val $conditioning_mode
        else
          sbatch launchers/train_ohe_protocol_axes.sbatch $rep $protocol_axis $unique_val $conditioning_mode
        fi
      done
    done
  done
done
