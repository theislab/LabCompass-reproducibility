#!/bin/bash

ENV_NAME="sc_exp_design"

# Parse Flags
case $1 in
-e|--env-name)
    ENV_NAME="$2"
    shift 2
    ;;
*)
    echo "Unknown parameter passed: $1"
    exit 1
    ;;
esac

echo "Conda environment set to ${ENV_NAME}"


BASE_SBATCH_DIR="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/sbatch"

# define experimental grid
TARGET_CELL_TYPES=(
    "CD14-Mono"
    # "CD49c-Myeloid"
    "CyclingProgenitor*"
    "DCs-CD14"
    # "EarlyGMP"
    "Early_Pro-B"
    "EosBasoMastPre"
    # "EosBasoMastPre_2"
    "EryPro"
    # "GMP-Cycle"
    # "GMP-Neutro"
    "GMP-Neutro/CD16-Mono"
    "HSCs"
    # "MEP"
    # "MLP"
    # "MPP"
    "MgkPro"
    "Pro-B"
    # "pDCs/cDCs"
)
SCHEDULERS=(
    "constant"
    "exp-decay"
    "reciprocal"
    "lin-decay"
)
OPTIMIZATION_TYPE=(
    # "unconstrained"
    # "penalized_oxy_days"
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
                  --query-type ${query_type} \
                  --env-name ${ENV_NAME};
            done
        done
    done
done
