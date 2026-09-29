#!/bin/bash
#SBATCH --job-name=inverse_plot
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint="a100_80gb|h100_80gb"

# SLURM copies the batch script to a node-local spool dir, so BASH_SOURCE does
# not point into the repo. Recover the submitted path from the job record.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}' | awk '{print $1}')"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

if [ ! -d "$REPO_ROOT/scripts" ]; then
    echo "ERROR: REPO_ROOT resolved to '$REPO_ROOT', which is not the repository." >&2
    exit 1
fi


EXPERIMENT=$1

# Solution path 
EXPERIMENT_DIR=$REPO_ROOT/project_folder/output/inverse/loss_guidance/bloodplus_inverse_fm_official/

# Working directory
WDIR=$REPO_ROOT

cd "$WDIR" || exit 1

source ~/.bashrc
conda activate sc_exp_design_12

echo "Running experiment: $EXPERIMENT"
echo "Experiment dir: $EXPERIMENT_DIR"

python scripts/inverse/uncertainty/uncertainty_estimation.py \
    --target_cell_types "late_MgkPro" \
    --n_noise_samples 5000 \
    --n_populations 10 \
    --base_config_path ../../../inverse/loss_guidance/config/ \
    --base_config_name run_inverse \
    --result_dir "$EXPERIMENT_DIR" \
    --experiment_type "$EXPERIMENT" \
    --true_concentration_path $REPO_ROOT/project_folder/output/miscellaneous/unique_concentrations_adata_bloodplus.h5ad \
    --paths bloodplus_loop2 \
    --cell_type_column cell_type_leiden \
    --destination_file_name candidates_with_uncertainties_next_loop.csv