#!/bin/bash

#SBATCH --job-name=run_sensitivity_analysis
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions/sensitivity_analysis_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions/sensitivity_analysis_%j.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=500G
#SBATCH --time=10:00:00

grid_size={$1:-10}

echo "Executing script with parameters"
echo "PY_SCRIPT: ${PY_SCRIPT}"
echo "TARGET_CT: ${TARGET_CT}"
echo "RUN_ID: ${RUN_ID}"
echo "EXP_TYPE: ${EXP_TYPE}"
echo "BASE_DIR: ${BASE_DIR}"
echo "BASE_DIR: ${BASE_DIR}"
echo "grid_size: ${grid_size}"


conda run -n sc_exp_design --no-capture-output \
    python "$PY_SCRIPT" \
        --target_cell_type "$TARGET_CT" \
        --run "$RUN_ID" \
        --experiment_type "$EXP_TYPE" \
        --base_dir "$BASE_DIR" \
        --grid_size $grid_size
