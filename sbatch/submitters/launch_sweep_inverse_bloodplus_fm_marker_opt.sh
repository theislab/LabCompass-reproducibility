#!/bin/bash

ENV_NAME="sc_exp_design"
PATHS="bloodplus"
USE_FM=1
ENV_MANAGER="micromamba"

# Parse Flags
case $1 in
-e|--env-name)
    ENV_NAME="$2"
    shift 2
    ;;
-p|--paths)
    PATHS="$2"
    shift 2
    ;;
-f|--use-fm)
    USE_FM="$2"
    shift 2
    ;;
esac

# set paths for flow map
if [[ $USE_FM -eq 1 ]]; then
    PATHS="${PATHS}_fm"
fi

echo "Conda environment set to ${ENV_NAME}"

BASE_SBATCH_DIR="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/sbatch"

# define experimental grid
TARGET_MARKERS=(
    "CD41a",
    "CD56",
    "EPCR"
)

SCHEDULERS=(
    "constant"
    # "exp-decay"
    "reciprocal"
    # "lin-decay"
)
OPTIMIZATION_TYPE=(
    "unconstrained"
    "penalized_oxy_days"
    "penalized_all_axes"
)

# iterate over each axes
for target_marker in ${TARGET_MARKERS[@]}; do
    for scheduler in ${SCHEDULERS[@]}; do
        for optimization_type in ${OPTIMIZATION_TYPE[@]}; do

            # log submission arguments
            echo "# -------------------------------- #"
            echo "Launching experiment"
            echo "target_marker=${target_marker}"
            echo "scheduler=${scheduler}"
            echo "optimization_type=${optimization_type}"

            # submit job
            sbatch ${BASE_SBATCH_DIR}/sbatch_launchers/inverse/sweep/sweep_inverse.sbatch \
                --target-marker ${target_marker} \
                --target-ct ${TARGET_CELL_TYPE} \
                --scheduler ${scheduler} \
                --optimization-type ${optimization_type} \
                --env-name ${ENV_NAME} \
                --paths ${PATHS};
        done
    done
done
