#!/bin/bash
#SBATCH --job-name=inverse_plot_per_index
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --array=1-299

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

WDIR=$REPO_ROOT/inverse/experiments

source ~/.bashrc
conda activate sc_exp_design

cd "$WDIR" || exit 1

JOBS="${WDIR}/jobs_selected.txt"

echo $JOBS

# Get the line corresponding to this array task
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")

if [ -z "$LINE" ]; then
    echo "ERROR: Could not read line ${SLURM_ARRAY_TASK_ID} from $JOBS"
    exit 1
fi

TARGET_CT=$(echo "$LINE" | awk '{print $1}')
RUN=$(echo "$LINE" | awk '{print $2}')
IDX=$(echo "$LINE" | awk '{print $3}')

echo "Running target_ct=$TARGET_CT run=$RUN idx=$IDX"

python "${WDIR}/run_plots_inverse_with_idx.py" \
    --target_ct "$TARGET_CT" \
    --run "$RUN" \
    --idx "$IDX"
