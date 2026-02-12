#!/bin/bash

BASE_SBATCH_DIR="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/sbatch"

# define experimental grid
TARGET_CELL_TYPES=(
    "EryPro"
    "HSCs"
    "MgkPro"
)
SCHEDULERS=(
    "constant"
    "exp-decay"
    "reciprocal"
    "lin-decay"
)
OPTIMIZATION_TYPE=(
    "unconstrained"
    "penalized_oxy_days"
    "penalized_all_axes"
)
QUERY_TYPE=(
    "pure_populations"
    "custom_populations"
    "custom_populations_masked"
)

# iterate over each axes
for target_cell_type in ${TARGET_CELL_TYPES[@]}; do
    for scheduler in ${SCHEDULERS[@]}; do
        for optimization_type in ${OPTIMIZATION_TYPE[@]}; do
            for query_type in ${QUERY_TYPE[@]}; do

                # log submission arguments
                echo "# -------------------------------- #"
                echo "Launching experiment"
                echo "target_cell_type=${target_cell_type}"
                echo "scheduler=${scheduler}"
                echo "optimization_type=${optimization_type}"
                echo "query_type=${query_type}"

                # submit job
                sbatch ${BASE_SBATCH_DIR}/sbatch_launchers/inverse/sweep/sweep_inverse.sbatch \
                  --target-ct ${target_cell_type} \
                  --scheduler ${scheduler} \
                  --optimization-type ${optimization_type} \
                  --query-type ${query_type};
            done
        done
    done
done
