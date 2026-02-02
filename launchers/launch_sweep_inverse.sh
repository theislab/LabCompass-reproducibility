#!/bin/bash

TARGET_CELL_TYPES=(
    # "CD14-Mono"
    # "CD49c-Myeloid"
    # "CyclingProgenitor*"
    # "DCs-CD14"
    # "EarlyGMP"
    # "Early_Pro-B"
    # "EosBasoMastPre"
    # "EosBasoMastPre_2"
    "EryPro"
    # "GMP-Cycle"
    # "GMP-Neutro"
    # "GMP-Neutro/CD16-Mono"
    "HSCs"
    # "MEP"
    # "MLP"
    # "MPP"
    "MgkPro"
    # "Pro-B"
    # "pDCs/cDCs"
)
SCHEDULERS=(
    "constant"
    "exp-decay"
    "reciprocal"
    "lin-decay"
)
for target_cell_type in ${TARGET_CELL_TYPES[@]}; do
    for scheduler in ${SCHEDULERS[@]}; do
        echo "launch eperiment for CT: ${target_cell_type} SCHED: ${scheduler}"
        sbatch launchers/sweep_inverse.sbatch $target_cell_type $scheduler;
    done
done
