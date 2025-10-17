#!/bin/bash

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
)

for sample_rep in "${!SAMPLE_REPS[@]}"; do
  for protocol_axis in "${!PROTOCOL_AXES_TO_NUNIQUE[@]}"; do
    rep=${SAMPLE_REPS[$sample_rep]}
    nunique=${PROTOCOL_AXES_TO_NUNIQUE[$protocol_axis]}
    for unique_val in $(seq 0 $((nunique - 1))); do
    # for unique_val in "$unique_ids"; do
        echo "Using sample rep ${rep}"
        echo "Running for protocol axis: $protocol_axis with unique IDs: $unique_val"
        sbatch launchers/sweep_ohe_protocol_axes.sbatch $sample_rep $protocol_axis $unique_val
    done
  done
done
