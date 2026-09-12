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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

EXPERIMENT=$1

# EXPERIMENT_DIR=$REPO_ROOT/project_folder/output/inverse/loss_guidance/raw_data
EXPERIMENT_DIR=$REPO_ROOT/project_folder/output/inverse/loss_guidance/new_runs_mgkpro

WDIR=$REPO_ROOT

cd "$WDIR" || exit 1

source ~/.bashrc
conda activate sc_exp_design_12

echo "Running experiment: $EXPERIMENT"
echo "Experiment dir: $EXPERIMENT_DIR"

python scripts/inverse/uncertainty/uncertainty_estimation.py \
    --target_cell_types "CD14-Mono__CyclingProgenitor*__DCs-CD14__Early_Pro-B__EosBasoMastPre__EryPro__GMP-Neutro/CD16-Mono__HSCs__MgkPro__Pro-B" \
    --n_noise_samples 5000 \
    --n_populations 10 \
    --base_config_path ../../../inverse/loss_guidance/config/ \
    --base_config_name run_inverse \
    --result_dir "$EXPERIMENT_DIR" \
    --experiment_type "$EXPERIMENT" \
    --true_concentration_path $REPO_ROOT/project_folder/output/miscellaneous/unique_concentrations_adata.h5ad \
    --paths new_measurements