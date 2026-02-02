#!/bin/bash
#SBATCH --job-name=inverse_plot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --array=1-1383

WDIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/inverse/experiments

source ~/.bashrc
conda activate sc_exp_design

cd "$WDIR" || exit 1

JOBS="${WDIR}/jobs_selected.txt"

# Get the line corresponding to this array task
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")

if [ -z "$LINE" ]; then
    echo "ERROR: Could not read line ${SLURM_ARRAY_TASK_ID} from $JOBS"
    exit 1
fi

TARGET_CT=$(echo "$LINE" | awk '{print $1}')
RUN=$(echo "$LINE" | awk '{print $2}')

echo "Running target_ct=$TARGET_CT run=$RUN"

python "${WDIR}/run_plots_inverse.py" \
    --target_ct "$TARGET_CT" \
    --run "$RUN"
