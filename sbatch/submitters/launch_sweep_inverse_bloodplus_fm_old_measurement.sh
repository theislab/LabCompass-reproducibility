#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="sc_exp_design_12"
PATHS="bloodplus_fm_old_measurements"
USE_FM=0
ENV_MANAGER="conda"
# ENV_MANAGER="micromamba"
# PATHS="new_measurements"

# Parse Flags
while [[ $# -gt 0 ]]; do
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
  *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

# set paths for flow map
if [[ $USE_FM -eq 1 ]]; then
    PATHS="${PATHS}_fm"
fi

echo "Conda environment set to ${ENV_NAME}"
echo "Paths set to ${PATHS}"

BASE_SBATCH_DIR="$REPO_ROOT/sbatch"

# define experimental grid
TARGET_CELL_TYPES=(
    # "CD14Mono"
    # "CD16Mono"
    # "CyclingPro1"
    # "CyclingPro2"
    # "EosBasMast1"
    # "EosBasMast2"
    # "EosBasMast3"
    # "EryPro"
    # "HSCs"
    # "ProB"
    "late_MgkPro"
    # 'GMP-Neutro'
    # 'preProB'
    # 'pre_cDC1'
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
QUERY_TYPE=(
    "pure_populations"
    # "custom_populations"
    # "custom_populations_masked"
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
                  --env-name ${ENV_NAME} \
                  --paths ${PATHS};
            done
        done
    done
done
