#!/bin/bash

# ---- 1. Define constants to override with cli ----
# ENV_NAME="sc_exp_design"
ENV_NAME="sc_exp_design_12"
# ENV_MANAGER="micromamba"
ENV_MANAGER="conda"
N_SAMPLES=50 
N_FWD_SAMPLES=350
LOOP="3"
OLD_CLF="false"


# ---- 2. Parse Flags ----
case $1 in
-e|--env-name)
    ENV_NAME="$2"
    shift 2
    ;;
-em|--env-manager)
    ENV_MANAGER="$2"
    shift 2
    ;;
-n|--n-samples)
    N_SAMPLES="$2"
    shift 2
    ;;
-nf|--n-fwd-samples)
    N_FWD_SAMPLES="$2"
    shift 2
    ;;
-l|--loop)
    LOOP="$2"
    shift 2
    ;;
-ocl|--old-clf)
    OLD_CLF="true"
    shift 1
    ;;
esac


# ---- 3. Define experimental grid for target cell type ----
TARGET_CELL_TYPES=(
    # "CD14Mono"
    # "CD16Mono"
    # "CyclingPro1"
    # "CyclingPro2"
    # "EosBasMast1"
    # "EosBasMast2"
    # "EosBasMast3"
    "EryPro"
    # "HSCs"
    # "ProB"
    "late_MgkPro"
    # 'GMP-Neutro'
    # 'preProB'
    # 'pre_cDC1'
)


# ---- 4. Define experimental grid for schedulers ----
SCHEDULERS=(
    "constant"
    # "exp-decay"
    "reciprocal"
    # "lin-decay"
)


# ---- 5. Define experimental grid for optimization type ----
OPTIMIZATION_TYPE=(
    "unconstrained"
    "penalized_oxy_days"
    "penalized_all_axes"
)


# ---- 6. Define experimental grid for query type ----
QUERY_TYPE=(
    "pure_populations"
    # "custom_populations"
    # "custom_populations_masked"
)

# ---- 7. Define base sbatch directory ----
BASE_SBATCH_DIR="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/sbatch"

# ---- 8. iterate over each axes ----
for target_cell_type in ${TARGET_CELL_TYPES[@]}; do
    for scheduler in ${SCHEDULERS[@]}; do
        for optimization_type in ${OPTIMIZATION_TYPE[@]}; do
            for query_type in ${QUERY_TYPE[@]}; do

                # ----- 9. Log submission arguments ----
                echo "# -------------------------------- #"
                echo "Launching experiment"
                echo "target_cell_type=${target_cell_type}"
                echo "scheduler=${scheduler}"
                echo "optimization_type=${optimization_type}"
                echo "query_type=${query_type}"
                echo "old classifier=${OLD_CLF}"

                # ----- 10. Submit job ----
                if [[ "${OLD_CLF}" == "true" ]]; then
                    sbatch ${BASE_SBATCH_DIR}/sbatch_launchers/inverse/sweep/sweep_inverse_loops.sbatch \
                    --target-ct ${target_cell_type} \
                    --scheduler ${scheduler} \
                    --optimization-type ${optimization_type} \
                    --query-type ${query_type} \
                    --n-samples ${N_SAMPLES} \
                    --n-fwd-samples ${N_FWD_SAMPLES} \
                    --loop ${LOOP} \
                    --env-manager ${ENV_MANAGER} \
                    --env-name ${ENV_NAME} \
                    --old-clf;
                else
                    sbatch ${BASE_SBATCH_DIR}/sbatch_launchers/inverse/sweep/sweep_inverse_loops.sbatch \
                    --target-ct ${target_cell_type} \
                    --scheduler ${scheduler} \
                    --optimization-type ${optimization_type} \
                    --query-type ${query_type} \
                    --n-samples ${N_SAMPLES} \
                    --n-fwd-samples ${N_FWD_SAMPLES} \
                    --loop ${LOOP} \
                    --env-manager ${ENV_MANAGER} \
                    --env-name ${ENV_NAME};
                fi
            done
        done
    done
done
