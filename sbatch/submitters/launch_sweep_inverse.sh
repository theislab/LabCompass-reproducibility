#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="labcompass"
PATHS="bloodplus"
# PATHS="new_measurements"
DUMP_NAME="bloodplus_inverse_fm_loop3"

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
  -d|--dump-name)
      DUMP_NAME="$2"
      shift 2
      ;;
  *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

echo "Conda environment set to ${ENV_NAME}"
echo "Paths set to ${PATHS}"
echo "Dump name set to ${DUMP_NAME}"


BASE_SBATCH_DIR="$REPO_ROOT/sbatch"

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
    # "GMP-Neutro/CD16-Mono"
    "HSCs"
    # "MEP"
    # "MLP"
    # "MPP"
    # "MgkPro
    # "Pro-B"
    # "pDCs/cDCs"
)

SCHEDULERS=(
    "constant"
    "exp-decay"
    "reciprocal"
    "lin-decay"
)
OPTIMIZATION_TYPE=(
    "unconstrained"
    # "penalized_oxy_days"
    # "penalized_all_axes"
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
                  --paths ${PATHS} \
                  --dump-name ${DUMP_NAME};
            done
        done
    done
done
