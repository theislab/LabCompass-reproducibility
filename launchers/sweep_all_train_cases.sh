#!/bin/bash

# This script runs all combinations of sample reps and condition modes
# using the main train_flow.sh sbatch script

SAMPLES=("channel" "pca" "scatter" "channel_concat" "pca_concat")
CONDITIONS=("unconditional" "protocol_concat" "ohe_protocols" "ohe_protocols_axes")

for sample in "${SAMPLES[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        echo "Submitting job for sample: $sample, condition: $cond"
        sbatch launchers/sweep_cfm_per_case.sbatch "$sample" "$cond"
    done
done
