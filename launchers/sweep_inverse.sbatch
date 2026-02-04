#!/bin/bash

#SBATCH --job-name=sweep_inverse
#SBATCH --output=../logs/sweep_inverse_haematopoiesis.log
#SBATCH --error=../logs/sweep_inverse_haematopoiesis.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

TARGET_CELL_TYPE=${1:-"HSCs"}
SCHEDULER=${2:-"constant"}


if [[ "$SCHEDULER" == "constant" ]]; then
    hparam_search="sweep_constant"
elif [[ "$SCHEDULER" == "exp-decay" ]]; then
    hparam_search="sweep_exp_decay"
elif [[ "$SCHEDULER" == "reciprocal" ]]; then
    hparam_search="sweep_reciprocal"
elif [[ "$SCHEDULER" == "lin-decay" ]]; then
    hparam_search="sweep_lin_decay"
else
    echo "❌ Unknown scheduler: $SCHEDULER"
    exit 1
fi

USE_NEW_QUERY=0
paths="default"
script_name="run_inverse_experiment"
if [[ USE_NEW_QUERY -eq 1 ]]; then
    paths="new_queries"
    script_name="run_inverse_experiment_new_targets"
fi
echo "cell type: ${TARGET_CELL_TYPE}"
echo "scheduler: ${SCHEDULER}"
echo "hparam_search: ${hparam_search}"
echo "paths: ${paths}"
echo "script_name: ${script_name}"

export HYDRA_FULL_ERROR=1

conda run -n sc_exp_design --no-capture-output \
    python ./inverse/experiments/${script_name}.py \
    --multirun \
    +launcher=slurm \
    +hparam_search="${hparam_search}" \
    paths=${paths} \
    sampling.target_cell_type='"'"${TARGET_CELL_TYPE}"'"' \
    2>&1 | tee "../logs/sweep_inverse-${TARGET_CELL_TYPE}_${SCHEDULER}.out"
